import pickle as pk

import pandas as pd
from loguru import logger
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor

from config import model_settings
from model.pipeline.preparation import prepare_data


def build_model() -> None:
    logger.info('Staring model build pipeline')
    df = prepare_data()

    feature_names = [
        'area',
        'constraction_year',
        'bedrooms',
        'garden',
        'balcony_yes',
        'parking_yes',
        'furnished_yes',
        'garage_yes',
        'storage_yes',
    ]

    X, y = _get_x_y(
        df,
        col_x=feature_names,
    )

    X_train, X_test, y_train, y_test = _split_train_test(
        X,
        y,
    )

    rf = _train_model(
        X_train,
        y_train,
    )
    _evaluation_model(
        rf,
        X_test,
        y_test,
    )

    _save_model(rf)


def _get_x_y(
        dataframe: pd.DataFrame,
        col_x: list[str],
        col_y: str = 'rent',
) -> tuple[pd.DataFrame, pd.Series]:

    logger.info(f'Defining X and y Variable, X var {col_x}, y var {col_y}')
    return dataframe[col_x], dataframe[col_y]


def _split_train_test(
        feature: pd.DataFrame,
        target: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    logger.info('Spliting into data train and test sets')
    X_train, X_test, y_train, y_test = train_test_split(
        feature,
        target,
        test_size=0.2,  # noqa: WPS432
    )
    return X_train, X_test, y_train, y_test


def _train_model(
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
) -> RandomForestRegressor:

    logger.info('Training Model with Hyperparameter -> RandomForestRegressor')
    grid_space = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 9, 12]
    }

    logger.debug(f'Grid space : {grid_space}')
    grid = GridSearchCV(
        RandomForestRegressor(),
        param_grid=grid_space,
        cv=5,
        scoring='r2',
    )
    model_grid = grid.fit(
        X_train,
        y_train,
    )
    return model_grid


def _evaluation_model(
        model: RandomForestRegressor,
        X_test: pd.DataFrame,
        y_test: pd.Series,
) -> float:

    logger.info(f'Evaluation model Sore : {model.score(X_test, y_test)}')
    return model.score(X_test, y_test)


model_path = f'{model_settings.model_path}/{model_settings.model_name}'


def _save_model(model: RandomForestRegressor) -> None:
    logger.info(f'Saving a model at directory:{model_path}')

    with open(model_path, 'wb') as model_file:
        pk.dump(model, model_file)
