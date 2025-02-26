from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict


class LoggerSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='config/.env',
        env_file_encode='utf-8',
        extra='ignore',
    )

    log_level: str


def configure_logger(log_level):
    logger.remove()
    logger.add(
        'logs/app.log',
        rotation='1 day',
        retention='2 days',
        compression='zip',
        level=log_level,
    )


configure_logger(LoggerSettings().log_level)
