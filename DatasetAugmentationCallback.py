from tf_keras_safe_imports.callbacks import Callback


# Augment the dataset at the end of every epoch
class DatasetAugmentationCallback(Callback):
    def __init__(self, abc_model):
        self.abc_model = abc_model

    def on_epoch_end(self, epoch, logs=None):
        self.abc_model.augment_training_dataset()
