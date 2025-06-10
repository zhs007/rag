from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    chroma_db_path: str = "./chroma_db"
    gemini_api_key: str
    gemini_model: str = "gemini-2.5-flash-preview-05-20"
    data_dir: str = "./app/data"
    env: str = "dev"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
