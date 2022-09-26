from tensorflow.keras.layers import Layer, Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import Model
import tensorflow as tf
import numpy as np


class GeneratePatches(Layer):
    """
    A layer for generating the image patches.
    """

    def __init__(self, patch):
        super(GeneratePatches, self).__init__()
        self.patch = patch

    def call(self, images):
        """
        Generate the images patch and flatten them
        :param images: batch of images
        :return:
        """

        images_patches = tf.image.extract_patches(images=images,
                                                  sizes=[1, self.patch, self.patch, 1],
                                                  strides=[1, self.patch, self.patch, 1],
                                                  rates=[1, 1, 1, 1],
                                                  padding='VALID')

        patch_dimension = images_patches.shape[-1]
        N = images_patches.shape[1] * images_patches.shape[2]

        patches = tf.reshape(images_patches, [-1, N, patch_dimension])

        return patches


class PositionalEncoding(Layer):
    """
    Implementation of the positional encoding layer
    """

    def __init__(self, d_model, n_positions):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.n_positions = n_positions

    def __get_angles(self, i, positions):
        """
        Get the angle for each position.
        :return:
        """

        angles = positions / np.power(10000, 2 * (i//2) / np.float(self.d_model))

        return angles

    def call(self, X):
        """
        Implements the positional encoding layer. In the first step, the positional encoding are computed.
        Finally, it adds to the  X
        :param X: the input tensor of the positional encoding layer
        :return:
        """

        positional_encoding = np.zeros(shape=(self.n_positions, self.d_model))
        positions = np.arange(self.n_positions)[:, np.newaxis]
        i = np.arange(self.d_model)[np.newaxis, :]

        angles = self.__get_angles(i=i, positions=positions)

        positional_encoding[:, 0::2] = np.sin(angles[:, 0::2])
        positional_encoding[:, 1::2] = np.cos(angles[:, 1::2])

        # self.positional_encoding = self.positional_encoding[np.newaxis, ...]
        positional_encoding = tf.cast(positional_encoding, dtype=tf.float32)

        X = positional_encoding + X

        return X


class ViT:
    """
    Implementation of the ViT model for image classification.
    """

    def __init__(self, input_shape, classes, patch, d_model, n_positions):
        self.input_shape = input_shape
        self.classes = classes
        self.patch = patch
        self.d_model = d_model
        self.n_positions = n_positions

        self.generate_patches = GeneratePatches(patch=self.patch)
        self.positional_encoding = PositionalEncoding(d_model=self.d_model, n_positions=self.n_positions)

    def __call__(self):
        """
        Implement the ViT model
        :return:
        """

        # Define the input as a tensor with shape input_shape
        X_input = Input((self.input_shape[0], self.input_shape[1], 3))

        # Generate patches
        X = self.generate_patches(X_input)

        # Linear projection
        X = Dense(units=1024)(X)

        # Learnable token
        X_class = Dense(units=1024)(X)

        # Add positional encodings
        X = self.positional_encoding(X)

        model = Model(inputs=X_input, outputs=X)

        return model
