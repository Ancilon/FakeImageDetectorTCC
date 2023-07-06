from abc import ABC, abstractmethod
from EnvironmentUtils import EnvironmentUtils
import random
import tensorflow as tf

# IDE safe imports
from tf_keras_safe_imports.utils import image_dataset_from_directory


class ABCModel(ABC):
    train_split = 0.7
    val_split = 0.2
    test_split = 0.1
    image_size = (224, 224)  # 224, 224 and 576, 324 and 432, 243 and 288, 162 experimented with
    seed = random.randint(0, 100000)
    batch_size = 32
    epochs = 500

    def __init__(self, dataset_path, es_callbacks):
        self.es_callbacks = es_callbacks
        self.dataset_path = dataset_path

        self.training_dataset = None
        self.validation_dataset = None
        self.test_dataset = None
        self.reload_datasets()

        self.model = None
        self.history = None
        self.create_model()

    def generate_training_dataset(self):
        train_ds = image_dataset_from_directory(
            self.dataset_path,
            labels="inferred",
            label_mode="binary",
            validation_split=ABCModel.train_split,
            subset="training",
            seed=ABCModel.seed,
            image_size=ABCModel.image_size,
            batch_size=ABCModel.batch_size,
        )
        return train_ds

    def generate_validation_dataset(self):
        val_ds = image_dataset_from_directory(
            self.dataset_path,
            labels="inferred",
            label_mode="binary",
            validation_split=ABCModel.val_split,
            subset="validation",
            seed=ABCModel.seed,
            image_size=ABCModel.image_size,
            batch_size=ABCModel.batch_size,
        )
        return val_ds

    def generate_test_dataset(self):
        test_ds = image_dataset_from_directory(
            self.dataset_path,
            labels="inferred",
            label_mode="binary",
            validation_split=ABCModel.test_split,
            subset="validation",
            seed=ABCModel.seed,
            image_size=ABCModel.image_size,
            batch_size=ABCModel.batch_size,
        )
        return test_ds

    def reload_datasets(self):
        EnvironmentUtils.set_tf_seed(ABCModel.seed)
        self.training_dataset = self.generate_training_dataset()
        self.validation_dataset = self.generate_validation_dataset()
        self.test_dataset = self.generate_test_dataset()

    @abstractmethod
    def create_model(self):
        pass

    def compile_model(self):
        self.model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

    def fit_model(self):
        # define steps per epoch and val steps
        steps_per_epoch = self.training_dataset.cardinality().numpy()
        val_steps = self.validation_dataset.cardinality().numpy()

        # Fitting the model
        history = self.model.fit(
            x=self.training_dataset,
            epochs=ABCModel.epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=self.validation_dataset,
            validation_steps=val_steps,
            callbacks=self.es_callbacks
        )
        self.history = history

    def load_model_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def evaluate_model_training(self):
        test_loss, test_acc = self.model.evaluate(self.training_dataset)
        print('Test accuracy:', test_acc, ' - test loss:', test_loss)

    def evaluate_model_validation(self):
        test_loss, test_acc = self.model.evaluate(self.validation_dataset)
        print('Test accuracy:', test_acc, ' - test loss:', test_loss)

    def evaluate_model_test(self):
        test_loss, test_acc = self.model.evaluate(self.test_dataset)
        print('Test accuracy:', test_acc, ' - test loss:', test_loss)

    # takes an image label pair and performs random changes in hopes of helping a model learn more use cases
    @staticmethod
    def augment_image(image, label):
        # Randomly flip the image horizontally or vertically
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)

        # Randomly adjust the brightness, contrast, saturation and hue
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
        image = tf.image.random_hue(image, max_delta=0.2)


        # Randomly crop the image
        # image = tf.image.random_crop(image, size=ABCModel.image_size + tf.shape(image)[-1])

        # Normalize the pixel values to the range [0, 1]
        image = image / 255.0

        return image, label

    # takes a tf.data.Dataset and performs random operations on its images to augment the dataset
    @staticmethod
    def augment_dataset(dataset):
        augmented_dataset = dataset.map(ABCModel.augment_image)
        return augmented_dataset

    def augment_training_dataset(self):
        self.training_dataset = ABCModel.augment_dataset(self.training_dataset)

    def augment_validation_dataset(self):
        self.validation_dataset = ABCModel.augment_dataset(self.validation_dataset)

    def augment_test_dataset(self):
        self.test_dataset = ABCModel.augment_dataset(self.test_dataset)

    def add_callback(self, callback):
        self.es_callbacks.append(callback)
