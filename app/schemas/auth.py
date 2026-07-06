from pydantic import BaseModel, Field


from typing import Optional

class AuthRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=80)
    password: str = Field(..., min_length=1, max_length=200)


class RegisterRequest(AuthRequest):
    password: str = Field(..., min_length=6, max_length=200)
    registration_number: Optional[str] = None
    programme: Optional[str] = None
    campus: Optional[str] = None
    admission_year: Optional[int] = None


class ChangePasswordRequest(BaseModel):
    user_id: int
    old_password: str = Field(..., min_length=1, max_length=200)
    new_password: str = Field(..., min_length=6, max_length=200)


class UserResponse(BaseModel):
    id: int
    username: str
    role: str = "user"
    registration_number: Optional[str] = None
    programme: Optional[str] = None
    campus: Optional[str] = None
    admission_year: Optional[int] = None

    class Config:
        from_attributes = True
