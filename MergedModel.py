import tensorflow as tf
from ModelTrainingUtils import ModelTrainingUtils
from ABCModel import ABCModel

# IDE safe imports
from tf_keras_safe_imports.layers import concatenate, Dense, BatchNormalization
from tf_keras_safe_imports.models import Model
from tf_keras_safe_imports.regularizers import L2


class MergedModel(ABCModel):
    def __init__(self, model_a, model_b, model_c, es_callbacks):
        self.model_a = model_a
        self.model_b = model_b
        self.model_c = model_c
        self.es_callbacks = es_callbacks

        self.model_a.reload_datasets()
        self.model_b.reload_datasets()
        self.model_c.reload_datasets()

        self.training_dataset = self.generate_training_dataset()
        self.validation_dataset = self.generate_validation_dataset()
        self.test_dataset = self.generate_test_dataset()
        self.model = None
        self.history = None
        self.model = self.create_model()
        ModelTrainingUtils.freeze_model(self.model, 9)

    @staticmethod
    def combine_datasets(gen1, gen2, gen3):
        zipp_joined_gen = tf.data.Dataset.zip((gen1, gen2, gen3))
        combined_dataset = zipp_joined_gen.map(lambda x1, x2, x3: ((x1[0], x2[0], x3[0]), x1[1]))
        return combined_dataset

    def generate_training_dataset(self):
        gen1 = self.model_a.training_dataset
        gen2 = self.model_b.training_dataset
        gen3 = self.model_c.training_dataset

        return self.combine_datasets(gen1, gen2, gen3)

    def generate_validation_dataset(self):
        gen1 = self.model_a.validation_dataset
        gen2 = self.model_b.validation_dataset
        gen3 = self.model_c.validation_dataset

        return self.combine_datasets(gen1, gen2, gen3)

    def generate_test_dataset(self):
        gen1 = self.model_a.test_dataset
        gen2 = self.model_b.test_dataset
        gen3 = self.model_c.test_dataset

        return self.combine_datasets(gen1, gen2, gen3)

    def create_model(self):
        # Concatenated Model of A B and C
        a_final_layer = self.model_a.model.get_layer(index=-2)
        b_final_layer = self.model_b.model.get_layer(index=-2)
        c_final_layer = self.model_c.model.get_layer(index=-2)

        a_b_c_concat = concatenate([a_final_layer.output, b_final_layer.output, c_final_layer.output],
                                   name="a_b_c_concatenated_layer")

        merged_dense_4 = Dense(32, activation="relu", kernel_regularizer=L2())(a_b_c_concat)
        merged_dense_4 = BatchNormalization()(merged_dense_4)

        # Final Layer
        output_layer_merged = Dense(1, activation='sigmoid', name="output_layer")(merged_dense_4)

        # Model Definition
        a_input_layer = self.model_a.model.input
        b_input_layer = self.model_b.model.input
        c_input_layer = self.model_c.model.input
        model_merged = Model(inputs=[a_input_layer, b_input_layer, c_input_layer], outputs=[output_layer_merged],
                             name="merged_model")

        return model_merged

    def fit_model(self):
        # define steps per epoch and val steps
        steps_per_epoch = self.model_a.training_dataset.cardinality().numpy()
        val_steps = self.model_a.validation_dataset.cardinality().numpy()

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
