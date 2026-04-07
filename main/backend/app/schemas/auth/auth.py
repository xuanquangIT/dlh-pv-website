import uuid
from datetime import datetime

from pydantic import BaseModel, EmailStr


class UserBase(BaseModel):
    username: str
    email: str
    full_name: str | None = None
    role_id: str


class UserCreate(UserBase):
    email: EmailStr
    password: str


class AdminUserCreate(UserCreate):
    is_active: bool = True


class UserRead(UserBase):
    id: uuid.UUID
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class LoginRequest(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserStatusUpdate(BaseModel):
    is_active: bool


class UserPasswordUpdate(BaseModel):
    new_password: str
