import numpy as np
import matplotlib.pyplot as plt
import visualkeras
from tensorflow import reduce_sum
from tf_keras_safe_imports.losses import MeanSquaredError
from sklearn.metrics import roc_curve, auc
from PIL import ImageFont
import tensorflow as tf

# IDE safe imports
from tf_keras_safe_imports.utils import plot_model


class ModelTrainingUtils:
    def __init__(self):
        pass

    @staticmethod
    # Show n images from the dataset along with their labels.
    # dataset (tf.data.Dataset): The dataset from which to display the images.
    # start (int): The index of the first image to display.
    # n (int): The number of images to display.
    # batch_size (int): The batch size of the dataset.
    # labels (list): A list of label strings corresponding to the labels of the images in the dataset.
    def show_images_from_dataset(dataset, start, n, batch_size, labels):
        total = n + start
        batches_to_take = total // batch_size + 1
        images_shown = 0

        # .take gets all images and labels in an epoch, -1 gets all epochs
        for images, images_labels in dataset.take(batches_to_take):
            for image, image_label in zip(images, images_labels):
                if images_shown >= total:
                    return
                elif images_shown < start:
                    images_shown += 1
                    continue
                plt.imshow(np.array(image.numpy(), np.int32))
                plt.title(f'N:{images_shown + 1} classification:{labels[image_label.numpy()]}')
                plt.show()
                images_shown += 1

    @staticmethod
    # Plot the training and validation metric accuracy over epochs in graph, shows then saves it.
    # model_history (keras.callbacks.History): The history object returned by the model.fit() method.
    # model_name (str): The name of the model being trained.
    # metric (str): The name of the metric to plot, either 'accuracy' or 'loss'.
    # folder_name (str): The folder to save the plotted metric.
    def plot_model_training_data(model_history, model_name, metric, folder_name):
        folder_name = folder_name if folder_name == "" else folder_name + '/'

        plotable_values = {"accuracy": "val_accuracy", "loss": "val_loss"}
        assert metric in plotable_values, "Metric must be 'accuracy' or 'loss'"

        plt.plot(model_history.history[metric])
        plt.plot(model_history.history[plotable_values[metric]])
        plt.title(model_name + " - " + metric)
        plt.ylabel(metric)
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(folder_name + model_name + "_" + metric + '.png', bbox_inches='tight')
        plt.close('all')

    # Freezes all layers in the model, except for the specified number of layers from the end.
    # n_layers_to_not_freeze (int): number of layers to remain trainable at the end of the model
    @staticmethod
    def freeze_model(model, n_layers_to_not_freeze):

        for layer in model.layers[:len(model.layers) - n_layers_to_not_freeze]:
            layer.trainable = False
        for layer in model.layers[len(model.layers) - n_layers_to_not_freeze:]:
            layer.trainable = True

    # Shows info about the model
    @staticmethod
    def model_summary(model):
        print(f"Model Layer Count:{len(model.layers)}")
        model.summary()

    # Calculates MSE of a model on the tf.data.Dataset object
    @staticmethod
    def calculate_mse(model, dataset, name):
        mse_loss = MeanSquaredError()  # Use tf.keras.losses.MeanSquaredError()

        mse = 0.0
        num_samples = 0

        for x, y in dataset:
            predictions = model(x, training=False)
            batch_mse = mse_loss(y, predictions)

            mse += tf.reduce_sum(batch_mse).numpy()  # Use tf.reduce_sum()

            num_samples += tf.shape(x)[0].numpy()  # Access the shape correctly

        mse /= num_samples

        print(f'{name} MSE on given dataset:{mse}')

        return mse

    # Plots the Receiver Operating Characteristic (ROC) curve for a given model on a given test tf.data.Dataset
    @staticmethod
    def plot_model_roc(model, dataset, model_name, folder_name):
        # Extract labels and predictions from the dataset
        folder_name = folder_name if folder_name == "" else folder_name + '/'

        true_labels = []
        predicted_probs = []

        for features, labels in dataset:
            predictions = model.predict(features)
            true_labels.extend(labels.numpy())
            predicted_probs.extend(predictions.flatten())

        # Compute false positive rate, true positive rate, and threshold
        fpr, tpr, thresholds = roc_curve(true_labels, predicted_probs)

        # Compute Area Under the Curve (AUC)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.savefig(folder_name + model_name + "ROC_Curve" + '.png', bbox_inches='tight')
        plt.close('all')

    @staticmethod
    # Plot the training and validation metric accuracy over epochs in graph, shows then saves it.
    # model_name (str): The name of the model being trained.
    # folder_name (str): The folder to save the plotted metric.
    def plot_model_architecture_visualkeras(model, model_name, folder_name):
        folder_name = folder_name if folder_name == "" else folder_name + '/'

        font = ImageFont.truetype("arial.ttf", 32)  # using comic sans is strictly prohibited!
        save_file_path = folder_name + model_name + "_Architecture" + "_Visualkeras" + '.png'
        visualkeras.layered_view(model, legend=True, font=font, to_file=save_file_path)  # font is optional!

    @staticmethod
    # Plot the training and validation metric accuracy over epochs in graph, shows then saves it.
    # model_name (str): The name of the model being trained.
    # folder_name (str): The folder to save the plotted metric.
    def plot_model_architecture_tensorflow(model, model_name, folder_name):
        folder_name = folder_name if folder_name == "" else folder_name + '/'

        plot_model(model, folder_name + model_name + "_Architecture" + "_Tensorflow" + '.png', show_shapes=True)





