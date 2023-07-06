from ABCModel import ABCModel

# IDE safe imports
from tf_keras_safe_imports.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from tf_keras_safe_imports.models import Model
from tf_keras_safe_imports.regularizers import L2, L1L2


class ModelB(ABCModel):

    def create_model(self):
        # -----------Model B
        b_ip_img = Input(shape=(ModelB.image_size + (3,)), name="Input_b")
        # 2 Layer
        b_conv_1 = Conv2D(64, (3, 3), kernel_regularizer=L2(), activation='relu')(b_ip_img)
        b_conv_1 = BatchNormalization()(b_conv_1)
        b_conv_1 = MaxPooling2D()(b_conv_1)
        # 2 Layer
        b_conv_2 = Conv2D(64, (3, 3), kernel_regularizer=L2(), activation='relu')(b_conv_1)
        b_conv_2 = BatchNormalization()(b_conv_2)
        b_conv_2 = MaxPooling2D()(b_conv_2)
        # 3 Layer
        b_conv_3 = Conv2D(128, (3, 3), kernel_regularizer=L2(), activation='relu')(b_conv_2)
        b_conv_3 = BatchNormalization()(b_conv_3)
        b_conv_3 = MaxPooling2D()(b_conv_3)
        # 4 Layer
        b_conv_4 = Conv2D(64, (2, 2), kernel_regularizer=L2(), activation='relu')(b_conv_3)
        b_conv_4 = BatchNormalization()(b_conv_4)
        b_conv_4 = MaxPooling2D()(b_conv_4)
        # 4 Layer
        b_conv_5 = Conv2D(32, (2, 2), kernel_regularizer=L2(), activation='relu')(b_conv_4)
        b_conv_5 = BatchNormalization()(b_conv_5)
        b_conv_5 = MaxPooling2D()(b_conv_5)

        a_cov_final = Flatten()(b_conv_5)

        # Dense layers
        b_dense_1 = Dense(64, kernel_regularizer=L2(), activation="relu", name="b_layer_1")(a_cov_final)
        b_dense_1 = BatchNormalization()(b_dense_1)
        # b_dense_1 = Dropout(0.5)(b_dense_1)
        b_dense_2 = Dense(128, kernel_regularizer=L2(), activation="relu", name="b_layer_2")(b_dense_1)
        b_dense_2 = BatchNormalization()(b_dense_2)
        # b_dense_2 = Dropout(0.5)(b_dense_2)
        b_dense_3 = Dense(64, kernel_regularizer=L2(), activation="relu", name="b_layer_3")(b_dense_2)
        b_dense_3 = BatchNormalization()(b_dense_3)
        # b_dense_3 = Dropout(0.5)(b_dense_3)
        b_dense_4 = Dense(32, kernel_regularizer=L2(), activation="relu", name="b_final_layer")(b_dense_3)
        b_dense_4 = BatchNormalization()(b_dense_4)
        # b_dense_4 = Dropout(0.5)(b_dense_4)

        # Output
        output_layer_b = Dense(1, activation='sigmoid', name="output_layer_b")(b_dense_4)

        model_b = Model(inputs=[b_ip_img], outputs=[output_layer_b], name="model_b")
        self.model = model_b
