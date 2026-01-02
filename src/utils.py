import os
import sys
import dill
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logger


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        logger.error("Error occurred in utils.save_object", exc_info=True)
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for model_name, model in models.items():
            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report

    except Exception as e:
        logger.error("Error occurred in utils.evaluate_models", exc_info=True)
        raise CustomException(e, sys)
