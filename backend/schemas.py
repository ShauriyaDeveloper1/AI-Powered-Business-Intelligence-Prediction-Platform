from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class LoginPayload(BaseModel):
    username: str
    password: str


class SignupPayload(BaseModel):
    username: str
    password: str
    confirm_password: str


class PredictPayload(BaseModel):
    task_type: str = Field(default="classification", pattern="^(classification|regression)$")
    upload_id: Optional[int] = None
    record: Dict[str, Any]
