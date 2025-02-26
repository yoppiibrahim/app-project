from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import DirectoryPath


class ModelSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='config/.env',
        env_file_encode='utf-8',
        extra='ignore',
    )

    model_path: DirectoryPath
    model_name: str


model_settings = ModelSettings()
