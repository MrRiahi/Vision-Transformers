from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
import tensorflow as tf
from math import prod

from src.ViT.ViT import ViT

from src.config import Config as Cfg


def get_model(classes_numbers):
    """
    This function builds the model defined in config.py file
    :param classes_numbers:
    :return:
    """

    if Cfg.MODEL_TYPE == 'ViT':
        # Build model
        input_shape = Cfg.VIT_SHAPE
        patch = Cfg.VIT_PATCH
        d_model = patch * patch * 3
        position = (input_shape[0] * input_shape[1]) // (patch ** 2)

        model = ViT(input_shape=Cfg.VIT_SHAPE,
                    classes=classes_numbers, patch=patch,
                    d_model=d_model, n_positions=position)()

        # Compile model
        optimizer = Adam(learning_rate=0.1)

        model.compile(loss=CategoricalCrossentropy(), optimizer=optimizer, metrics=['accuracy'])

    else:
        raise Exception('Invalid model type!')

    return model, input_shape


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
