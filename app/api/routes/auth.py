from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.security import (
    hash_password,
    is_password_hash,
    verify_legacy_plaintext_password,
    verify_password,
)
from app.core.config import settings
from app.db.models import User
from app.db.session import get_db
from app.schemas.auth import AuthRequest, RegisterRequest, UserResponse, ChangePasswordRequest

router = APIRouter()


def normalize_username(username: str) -> str:
    return username.strip().lower()


def get_user_role(username: str) -> str:
    admin_usernames = {
        normalize_username(admin_username)
        for admin_username in settings.ADMIN_USERNAMES.split(",")
        if admin_username.strip()
    }

    return "admin" if normalize_username(username) in admin_usernames else "user"


def serialize_user(user: User) -> UserResponse:
    return UserResponse(
        id=user.id,
        username=user.username,
        role=get_user_role(user.username),
        registration_number=user.registration_number,
        programme=user.programme,
        campus=user.campus,
        admission_year=user.admission_year,
    )


@router.post("/register", response_model=UserResponse, status_code=201)
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    username = normalize_username(payload.username)
    if not username:
        raise HTTPException(status_code=400, detail="Username is required")

    user = User(
        username=username,
        password_hash=hash_password(payload.password),
        registration_number=payload.registration_number,
        programme=payload.programme,
        campus=payload.campus,
        admission_year=payload.admission_year,
    )

    try:
        db.add(user)
        db.commit()
        db.refresh(user)
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="Username already exists")

    return serialize_user(user)


@router.post("/login", response_model=UserResponse)
def login(payload: AuthRequest, db: Session = Depends(get_db)):
    username = normalize_username(payload.username)
    user = db.query(User).filter(func.lower(User.username) == username).first()

    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    password_is_valid = verify_password(payload.password, user.password_hash)
    password_is_legacy_valid = verify_legacy_plaintext_password(payload.password, user.password_hash)

    if not password_is_valid and not password_is_legacy_valid:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    if password_is_legacy_valid or not is_password_hash(user.password_hash):
        user.password_hash = hash_password(payload.password)
        db.commit()
        db.refresh(user)

    return serialize_user(user)


@router.post("/change-password", response_model=dict)
def change_password(payload: ChangePasswordRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == payload.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    password_is_valid = verify_password(payload.old_password, user.password_hash)
    password_is_legacy_valid = verify_legacy_plaintext_password(payload.old_password, user.password_hash)

    if not password_is_valid and not password_is_legacy_valid:
        raise HTTPException(status_code=401, detail="Invalid old password")

    user.password_hash = hash_password(payload.new_password)
    db.commit()

    return {"message": "Password updated successfully"}
