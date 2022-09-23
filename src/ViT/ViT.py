from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Layer
import tensorflow as tf


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
                                                  padding='valid')

        batch_size = tf.shape(images)[0]
        patch_dimension = tf.shape(images_patches)[-1]

        patches = tf.reshape(images_patches, [batch_size, -1, patch_dimension])

        return patches


class ViT:
    """
    Implementation of the ViT model for image classification.
    """

    def __init__(self, classes, patch=1):
        self.classes = classes
        self.patch = patch

        self.generate_patches = GeneratePatches(patch=self.patch)

    def __call__(self, X):
        """
        Implement the ViT model
        :return:
        """

        # Define the input as a tensor with shape input_shape
        X_input = Input((self.input_shape[0], self.input_shape[1], 3))

