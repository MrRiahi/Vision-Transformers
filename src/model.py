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
        d_model = 1024
        d_mlp = 128
        n_encoder_layers = 2  # 4
        position = (input_shape[0] * input_shape[1]) // (patch ** 2) + 1

        model = ViT(input_shape=Cfg.VIT_SHAPE,
                    classes=classes_numbers, patch=patch,
                    d_model=d_model, n_positions=position,
                    d_mlp=d_mlp, n_layers=n_encoder_layers)

        # Compile model
        optimizer = Adam(learning_rate=3e-4)
        # optimizer = SGD(learning_rate=0.001)
        # model.compile(loss=CategoricalCrossentropy(), optimizer=optimizer, metrics=['accuracy'])

        model.compile(loss=CategoricalCrossentropy(), optimizer=optimizer, metrics=['accuracy'],
                      run_eagerly=False)

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
