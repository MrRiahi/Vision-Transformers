from tensorflow.keras.layers import Layer, Dense, LayerNormalization, MultiHeadAttention, Dropout
from tensorflow.keras.activations import gelu
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

    def call(self, X):
        """
        Generate the images patch and flatten them
        :param X: batch of images
        :return:
        """
        images_patches = tf.image.extract_patches(images=X,
                                                  sizes=[1, self.patch, self.patch, 1],
                                                  strides=[1, self.patch, self.patch, 1],
                                                  rates=[1, 1, 1, 1],
                                                  padding='VALID')

        # patch_dimension = images_patches.shape[-1]
        patch_dimensions = tf.shape(images_patches)
        # N = images_patches.shape[1] * images_patches.shape[2]
        N = patch_dimensions[1] * patch_dimensions[2]

        patches = tf.reshape(images_patches, [-1, N, patch_dimensions[-1]])

        return patches


class PositionalEncoding(Layer):
    """
    Implementation of the positional encoding layer
    """

    def __init__(self, d_model, n_positions):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.n_positions = n_positions
        self.class_embedding = self.add_weight('class_embedding', shape=(1, 1, self.d_model))

    def __get_angles(self, i, positions):
        """
        Get the angle for each position.
        :param i: position of each feature
        :param positions: position of each patch
        :return:
        """

        angles = positions / np.power(10000, 2 * (i//2) / np.float(self.d_model))

        return angles

    def __get_positional_encoding(self, angles, batch_size):
        """
        Compute the positional encoding based on the angles
        :param angles: angles for positional encoding
        :param batch_size: batch size
        :return:
        """
        positional_encoding = np.zeros(shape=(self.n_positions, self.d_model))

        positional_encoding[:, 0::2] = np.sin(angles[:, 0::2])
        positional_encoding[:, 1::2] = np.cos(angles[:, 1::2])
        positional_encoding = tf.cast(positional_encoding, dtype=tf.float32)
        positional_encoding = tf.broadcast_to(positional_encoding, shape=(batch_size,
                                                                          self.n_positions,
                                                                          self.d_model))

        return positional_encoding

    def __concat_class_embeddings_to_x(self, X, batch_size):
        """
        Get the class embeddings
        :param batch_size: batch size
        :param X: the input layer
        :return:
        """

        cls_embeddings = tf.broadcast_to(self.class_embedding, shape=(batch_size, 1, self.d_model))

        new_X = tf.concat([cls_embeddings, X], axis=1)

        return new_X

    def call(self, X):
        """
        Implements the positional encoding layer. In the first step, the positional encoding are computed.
        Finally, it adds to the  X
        :param X: the input tensor of the positional encoding layer
        :return:
        """

        batch_size = X.shape[0]
        positions = np.arange(self.n_positions)[:, np.newaxis]
        i = np.arange(self.d_model)[np.newaxis, :]

        angles = self.__get_angles(i=i, positions=positions)

        positional_encoding = self.__get_positional_encoding(angles=angles, batch_size=batch_size)

        X = self.__concat_class_embeddings_to_x(X=X, batch_size=batch_size)

        X = positional_encoding + X

        return X


class ViT(Model):
    """
    Implementation of the ViT model for image classification.
    """

    def __init__(self, input_shape, classes, patch, d_model, n_positions, d_mlp, n_layers):
        super(ViT, self).__init__()
        self.input_size = input_shape
        self.classes = classes
        self.patch = patch
        self.d_model = d_model
        self.n_positions = n_positions
        self.d_mlp = d_mlp
        self.n_layers = n_layers

        self.generate_patches = GeneratePatches(patch=self.patch)
        self.positional_encoding = PositionalEncoding(d_model=self.d_model, n_positions=self.n_positions)

    def __encoder(self, X_input, training):
        """
        Implementation of the encoder block of the ViT
        :param X_input:
        :param training:
        :return:
        """

        # Layer normalization
        X = LayerNormalization(epsilon=1e-6)(X_input)
        print(f'X shape is {tf.shape(X)}')
        print(f'X shape is {X.shape}')

        # Multi head attentions
        X = MultiHeadAttention(num_heads=3, key_dim=self.d_model)(X, X, X)

        # Dropout layer
        X = Dropout(rate=0.2)(X, training=training)

        # Residual layer
        X_res = X_input + X

        # Layer normalization
        X = LayerNormalization(epsilon=1e-6)(X_res)

        # Fully connected layer
        X = Dense(units=self.d_mlp, activation='relu')(X)
        X = Dense(units=self.d_model)(X)

        X = X + X_res

        return X

    def __mlp_head(self, X_class):
        """
        MLP for the head part of transformer.
        :param X_class:
        :return:
        """

        X = LayerNormalization(epsilon=1e-6)(X_class)
        X = Dense(units=self.d_mlp, activation='relu')(X)
        # X = gelu(X)
        X = Dropout(rate=0.2)(X)
        X_output = Dense(units=self.classes, activation='softmax')(X)

        return X_output

    def call(self, X, training):
        """
        Implement the ViT model
        :return:
        """

        # Generate patches
        X = self.generate_patches(X)

        # Linear projection
        X = Dense(units=1024)(X)

        # Add positional encodings
        X = self.positional_encoding(X)

        # Encoder part
        for _ in range(self.n_layers):
            X = self.__encoder(X_input=X, training=training)

        X = self.__mlp_head(X_class=X[:, 0, :])

        return X
