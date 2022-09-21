from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
import tensorflow as tf

from src.ViT import ViT

from src.config import Config as Cfg


def get_model(classes_numbers):
    """
    This function builds the model defined in config.py file
    :param classes_numbers:
    :return:
    """

    if Cfg.MODEL_TYPE == 'ViT':
        # Build model
        vit = ViT(classes=classes_numbers)
        model = vit()

        # Compile model
        optimizer = Adam(learning_rate=0.1)

        model.compile(loss=CategoricalCrossentropy(), optimizer=optimizer, metrics=['accuracy'])

    else:
        raise Exception('Invalid model type!')

    return model, input_size


def load_model(model_path):
    """
    This function loads the model in model_path
    :param model_path:
    :return:
    """

    if Cfg.MODEL_TYPE == 'ViT':
        input_shape = Cfg.ViT

        # Load model
        model = tf.keras.models.load_model(model_path)

    else:
        raise Exception('Invalid model type!')

    return model, input_shape
