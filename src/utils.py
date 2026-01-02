import os
import sys
import dill
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
