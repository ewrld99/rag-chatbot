import os
import sys

# Ensure the app module can be imported
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.db.session import SessionLocal
from app.db.models import DocumentModel

def cleanup():
    db = SessionLocal()
    try:
        # Get all distinct hashes
        hashes = db.query(DocumentModel.content_hash).filter(DocumentModel.content_hash.isnot(None)).distinct().all()
        
        deleted_count = 0
        deleted_files = 0
        
        for (h,) in hashes:
            if not h:
                continue
                
            # Find all docs with this hash, ordered by upload_date
            docs = db.query(DocumentModel).filter(DocumentModel.content_hash == h).order_by(DocumentModel.upload_date.asc()).all()
            
            if len(docs) > 1:
                # Keep the first one (the oldest)
                keep_doc = docs[0]
                delete_docs = docs[1:]
                
                print(f"Hash {h}: keeping {keep_doc.filename} ({keep_doc.id})")
                
                for doc in delete_docs:
                    print(f"  - Deleting duplicate DB record: {doc.filename} ({doc.id})")
                    db.delete(doc)
                    deleted_count += 1
                    
                    # Delete physical file from uploads
                    if doc.filename:
                        file_path = os.path.join("uploads", doc.filename)
                        if os.path.exists(file_path):
                            # Ensure we don't delete the original file
                            if doc.filename != keep_doc.filename:
                                os.remove(file_path)
                                deleted_files += 1
                                print(f"  - Deleted file: {file_path}")
                                
        db.commit()
        print(f"\nCleanup complete. Deleted {deleted_count} duplicate DB records and {deleted_files} physical files.")
        
    except Exception as e:
        print(f"Error during cleanup: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    cleanup()
