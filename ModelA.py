from ABCModel import ABCModel

# IDE safe imports
from tf_keras_safe_imports.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from tf_keras_safe_imports.models import Model
from tf_keras_safe_imports.regularizers import L2, L1L2


class ModelA(ABCModel):

    def create_model(self):
        # -----------Model A
        a_ip_img = Input(shape=(ModelA.image_size + (3,)), name="Input_a")
        # 2 Layer
        a_conv_1 = Conv2D(64, (3, 3), kernel_regularizer=L2(), activation='relu')(a_ip_img)
        a_conv_1 = BatchNormalization()(a_conv_1)
        a_conv_1 = MaxPooling2D()(a_conv_1)
        # 2 Layer
        a_conv_2 = Conv2D(64, (3, 3), kernel_regularizer=L2(), activation='relu')(a_conv_1)
        a_conv_2 = BatchNormalization()(a_conv_2)
        a_conv_2 = MaxPooling2D()(a_conv_2)
        # 3 Layer
        a_conv_3 = Conv2D(128, (3, 3), kernel_regularizer=L2(), activation='relu')(a_conv_2)
        a_conv_3 = BatchNormalization()(a_conv_3)
        a_conv_3 = MaxPooling2D()(a_conv_3)
        # 4 Layer
        a_conv_4 = Conv2D(64, (2, 2), kernel_regularizer=L2(), activation='relu')(a_conv_3)
        a_conv_4 = BatchNormalization()(a_conv_4)
        a_conv_4 = MaxPooling2D()(a_conv_4)
        # 4 Layer
        a_conv_5 = Conv2D(32, (2, 2), kernel_regularizer=L2(), activation='relu')(a_conv_4)
        a_conv_5 = BatchNormalization()(a_conv_5)
        a_conv_5 = MaxPooling2D()(a_conv_5)

        a_cov_final = Flatten()(a_conv_5)

        # Dense layers
        a_dense_1 = Dense(64, kernel_regularizer=L2(), activation="relu", name="a_layer_1")(a_cov_final)
        a_dense_1 = BatchNormalization()(a_dense_1)
        # a_dense_1 = Dropout(0.5)(a_dense_1)
        a_dense_2 = Dense(128, kernel_regularizer=L2(), activation="relu", name="a_layer_2")(a_dense_1)
        a_dense_2 = BatchNormalization()(a_dense_2)
        # a_dense_2 = Dropout(0.5)(a_dense_2)
        a_dense_3 = Dense(64, kernel_regularizer=L2(), activation="relu", name="a_layer_3")(a_dense_2)
        a_dense_3 = BatchNormalization()(a_dense_3)
        # a_dense_3 = Dropout(0.5)(a_dense_3)
        a_dense_4 = Dense(32, kernel_regularizer=L2(), activation="relu", name="a_final_layer")(a_dense_3)
        a_dense_4 = BatchNormalization()(a_dense_4)
        # a_dense_4 = Dropout(0.5)(a_dense_4)

        # Output
        output_layer_a = Dense(1, activation='sigmoid', name="output_layer_a")(a_dense_4)

        model_a = Model(inputs=[a_ip_img], outputs=[output_layer_a], name="model_a")
        self.model = model_a
