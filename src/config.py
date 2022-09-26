
class Config:

    VIT_SHAPE = (224, 224)

    VIT_PATCH = 32

    # 'ViT'
    MODEL_TYPE = 'ViT'

    MODEL_PATH = f'models/cifar-10/{MODEL_TYPE}'

    # Train config
    BUFFER_SIZE = 500
    BATCH_SIZE = 32
    # Each model trains for 300 epochs
    EPOCHS = 100

    TRAIN_SUBSET = 0.8
    VALIDATION_SUBSET = 1 - TRAIN_SUBSET

    TRAIN_DATASET_PATH = 'dataset/cifar-10/images/train'
    TEST_DATASET_PATH = 'dataset/cifar-10/images/test'

    # Dataset configs
    CIFAR_10_CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
                            'horse', 'ship', 'truck']
    CIFAR10_CLASS_NAME_TO_NUMBER = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3,
                                    'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
    CIFAR_10_CLASS_NUMBERS = 10
