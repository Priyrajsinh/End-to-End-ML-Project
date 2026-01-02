import os
import sys
from dataclasses import dataclass

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logger
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logger.info("Splitting training and test data")

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "XGBoost": XGBRegressor(
                    objective="reg:squarederror",
                    eval_metric="rmse"
                ),
                "CatBoost": CatBoostRegressor(verbose=False),
            }

            params = {
                "Linear Regression": {},
                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse"],
                },
                "Random Forest": {
                    "n_estimators": [50, 100, 200],
                },
                "Gradient Boosting": {
                    "learning_rate": [0.05, 0.1],
                    "n_estimators": [100, 200],
                },
                "AdaBoost": {
                    "learning_rate": [0.05, 0.1],
                    "n_estimators": [50, 100],
                },
                "XGBoost": {
                    "learning_rate": [0.05, 0.1],
                    "n_estimators": [100, 200],
                },
                "CatBoost": {
                    "depth": [6, 8],
                    "learning_rate": [0.05, 0.1],
                },
            }

            best_score = -1
            best_model = None
            best_model_name = None

            for model_name, model in models.items():
                logger.info(f"Tuning model: {model_name}")

                param_grid = params[model_name]

                if param_grid:
                    gs = GridSearchCV(
                        model,
                        param_grid,
                        cv=3,
                        scoring="r2",
                        n_jobs=-1
                    )
                    gs.fit(X_train, y_train)
                    tuned_model = gs.best_estimator_
                else:
                    tuned_model = model
                    tuned_model.fit(X_train, y_train)

                predictions = tuned_model.predict(X_test)
                score = r2_score(y_test, predictions)

                logger.info(f"{model_name} R2 score: {score}")

                if score > best_score:
                    best_score = score
                    best_model = tuned_model
                    best_model_name = model_name

            if best_score < 0.6:
                raise CustomException("No suitable model found", sys)

            logger.info(
                f"Best model: {best_model_name} with R2 score: {best_score}"
            )

            save_object(
                file_path=self.config.trained_model_file_path,
                obj=best_model
            )

            return best_score

        except Exception as e:
            logger.error("Error occurred in model trainer", exc_info=True)
            raise CustomException(e, sys)
