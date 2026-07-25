from typing import Any, Dict, List, Optional
import io
import logging
from sqlalchemy.orm import Session
from sqlalchemy import or_, desc
import pandas as pd
from datetime import datetime

from app.db.models import FAQModel, AuditLog, User
from app.services.embedding_service import get_embedding, get_embeddings

logger = logging.getLogger(__name__)

def log_audit(db: Session, setting_key: str, admin: str, new_value: Optional[str] = None, old_value: Optional[str] = None):
    audit = AuditLog(
        admin=admin or "Unknown",
        setting_key=setting_key,
        old_value=old_value,
        new_value=new_value
    )
    db.add(audit)


def _cell_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()
def get_admin_id(db: Session, admin_username: str) -> Optional[int]:
    if not admin_username or admin_username == "Unknown Admin":
        return None
    user = db.query(User).filter(User.username == admin_username).first()
    return user.id if user else None

class FAQService:
    def __init__(self, db: Session):
        self.db = db

    def get_faqs(
        self,
        skip: int = 0,
        limit: int = 100,
        search: Optional[str] = None,
        category: Optional[str] = None,
        is_active: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Fetch FAQs with pagination and filtering."""
        query = self.db.query(FAQModel)

        if search:
            search_term = f"%{search}%"
            query = query.filter(
                or_(
                    FAQModel.question.ilike(search_term),
                    FAQModel.answer.ilike(search_term),
                )
            )

        if category:
            query = query.filter(FAQModel.category == category)

        if is_active is not None:
            query = query.filter(FAQModel.is_active == is_active)

        total = query.count()
        faqs = query.order_by(desc(FAQModel.created_at)).offset(skip).limit(limit).all()

        # Extract unique categories
        categories = [
            row[0]
            for row in self.db.query(FAQModel.category).distinct().filter(FAQModel.category.isnot(None)).all()
        ]

        return {
            "items": faqs,
            "total": total,
            "skip": skip,
            "limit": limit,
            "categories": categories,
        }

    def get_faq(self, faq_id: str) -> Optional[FAQModel]:
        """Fetch a single FAQ by ID."""
        return self.db.query(FAQModel).filter(FAQModel.id == faq_id).first()

    def create_faq(self, question: str, answer: str, category: Optional[str] = None, admin_username: Optional[str] = None) -> FAQModel:
        """Create a new FAQ and generate its embedding."""
        embedding = get_embedding(f"Question: {question}\n\nAnswer: {answer}", self.db)
        admin_id = get_admin_id(self.db, admin_username)
        
        faq = FAQModel(
            question=question.strip(),
            answer=answer.strip(),
            category=category,
            embedding=embedding,
            created_by=admin_id,
        )
        
        self.db.add(faq)
        log_audit(self.db, "FAQ_CREATE", admin_username, new_value=question.strip())
        
        self.db.commit()
        self.db.refresh(faq)
        return faq

    def update_faq(self, faq_id: str, question: Optional[str] = None, answer: Optional[str] = None, category: Optional[str] = None, is_active: Optional[bool] = None, admin_username: Optional[str] = None) -> Optional[FAQModel]:
        """Update an existing FAQ. Regenerate embedding if question or answer changes."""
        faq = self.get_faq(faq_id)
        if not faq:
            return None
            
        needs_new_embedding = False
        old_question = faq.question
        
        if question is not None and question.strip() != faq.question:
            faq.question = question.strip()
            needs_new_embedding = True
            
        if answer is not None and answer.strip() != faq.answer:
            faq.answer = answer.strip()
            needs_new_embedding = True
            
        if category is not None:
            faq.category = category
            
        if is_active is not None:
            faq.is_active = is_active
            
        if needs_new_embedding:
            faq.embedding = get_embedding(f"Question: {faq.question}\n\nAnswer: {faq.answer}", self.db)
            
        log_audit(self.db, "FAQ_UPDATE", admin_username, old_value=old_question, new_value=faq.question)
        self.db.commit()
        self.db.refresh(faq)
        return faq

    def delete_faq(self, faq_id: str, admin_username: Optional[str] = None) -> bool:
        """Soft delete an FAQ."""
        faq = self.get_faq(faq_id)
        if not faq:
            return False
            
        faq.is_active = False
        log_audit(self.db, "FAQ_DELETE", admin_username, old_value=faq.question)
        self.db.commit()
        return True
        
    def bulk_import(self, file_content: bytes, filename: str, admin_username: Optional[str] = None) -> Dict[str, Any]:
        """Import FAQs from CSV or XLSX using batched embeddings."""
        stats = {
            "imported": 0,
            "skipped": 0,
            "duplicates": 0,
            "errors": 0,
            "error_details": []
        }
        
        try:
            if filename.lower().endswith('.csv'):
                df = pd.read_csv(io.BytesIO(file_content))
            elif filename.lower().endswith(('.xls', '.xlsx')):
                df = pd.read_excel(io.BytesIO(file_content))
            else:
                raise ValueError("Unsupported file format. Please upload CSV or Excel.")
                
            required_cols = ['Question', 'Answer']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
                
        except Exception as e:
            stats["errors"] += 1
            stats["error_details"].append(f"File parsing error: {str(e)}")
            return stats

        admin_id = get_admin_id(self.db, admin_username)
        valid_rows = []
        
        # Pre-processing and deduplication
        for index, row in df.iterrows():
            question = _cell_text(row.get('Question', ''))
            answer = _cell_text(row.get('Answer', ''))
            category = _cell_text(row.get('Category', '')) or None
                
            if not question or not answer:
                stats["skipped"] += 1
                stats["error_details"].append(f"Row {index + 2}: Missing question or answer")
                continue
                
            # Check for duplicate question
            existing = self.db.query(FAQModel).filter(FAQModel.question == question).first()
            if existing:
                stats["duplicates"] += 1
                stats["skipped"] += 1
                continue
                
            valid_rows.append({
                "row_num": index + 2,
                "question": question,
                "answer": answer,
                "category": category
            })
            
        # Batch Embedding processing
        batch_size = 50
        for i in range(0, len(valid_rows), batch_size):
            batch = valid_rows[i:i + batch_size]
            texts_to_embed = [f"Question: {r['question']}\n\nAnswer: {r['answer']}" for r in batch]
            
            try:
                embeddings = get_embeddings(texts_to_embed, self.db)
                new_faqs = []
                for j, item in enumerate(batch):
                    new_faqs.append(FAQModel(
                        question=item['question'],
                        answer=item['answer'],
                        category=item['category'],
                        embedding=embeddings[j],
                        created_by=admin_id
                    ))
                self.db.add_all(new_faqs)
                self.db.commit()
                stats["imported"] += len(batch)
            except Exception as e:
                self.db.rollback()
                logger.error(f"Error importing batch starting at row {batch[0]['row_num']}: {e}")
                stats["errors"] += len(batch)
                stats["error_details"].append(f"Rows {batch[0]['row_num']} to {batch[-1]['row_num']}: Embedding/DB failed")

        if stats["imported"] > 0:
            log_audit(self.db, "FAQ_IMPORT", admin_username, new_value=f"Imported {stats['imported']} FAQs from {filename}")
            self.db.commit()
                
        return stats
