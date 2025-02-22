# ⚠️ IMPORTANT
# This is proprietary code of Outhad AI (outhad.com) developed by Mohammad Tanzil Idrisi

from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    SQL_CONNECTION_STRING: Optional[str] = None
    NEO4J_URI: Optional[str] = None
    NEO4J_USER: Optional[str] = None
    NEO4J_PASSWORD: Optional[str] = None
    VECTOR_DIMENSION: int = 128
    
    class Config:
        env_file = ".env" 