import pickle as pk

from loguru import logger
from pathlib import Path

from config import model_settings
from model.pipeline.model import build_model


class ModelService():
    def __init__(self):
        self.model = None

    def load_model(self) -> None:
        logger.info(
            f'Checking the existence of model config file at '
            f'{model_settings.model_path}/{model_settings.model_name}',
        )

        model_path = Path(
            f'{model_settings.model_path}/{model_settings.model_name}'
        )

        if not model_path.exists():
            logger.info(
                f'Model at {model_path}'
                f'was not found, building {model_settings.model_name}',
            )
            build_model()

        logger.info(
            f'Model {model_settings.model_name} exist..!!'
            f'loading model configuration file',
        )

        with open(model_path, 'rb') as model_file:
            self.model = pk.load(model_file)

    def predict(self, input_parameters: list) -> list:
        logger.info('Make Prediction')
        return self.model.predict([input_parameters])
