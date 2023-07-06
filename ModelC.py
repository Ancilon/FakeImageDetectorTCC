from ABCModel import ABCModel

# IDE safe imports
from tf_keras_safe_imports.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from tf_keras_safe_imports.models import Model
from tf_keras_safe_imports.regularizers import L2, L1L2


class ModelC(ABCModel):

    def create_model(self):
        # -----------Model A
        c_ip_img = Input(shape=(ModelC.image_size + (3,)), name="Input_c")
        # 2 Layer
        c_conv_1 = Conv2D(64, (3, 3), kernel_regularizer=L2(), activation='relu')(c_ip_img)
        c_conv_1 = BatchNormalization()(c_conv_1)
        c_conv_1 = MaxPooling2D()(c_conv_1)
        # 2 Layer
        c_conv_2 = Conv2D(64, (3, 3), kernel_regularizer=L2(), activation='relu')(c_conv_1)
        c_conv_2 = BatchNormalization()(c_conv_2)
        c_conv_2 = MaxPooling2D()(c_conv_2)
        # 3 Layer
        c_conv_3 = Conv2D(128, (3, 3), kernel_regularizer=L2(), activation='relu')(c_conv_2)
        c_conv_3 = BatchNormalization()(c_conv_3)
        c_conv_3 = MaxPooling2D()(c_conv_3)
        # 4 Layer
        c_conv_4 = Conv2D(64, (2, 2), kernel_regularizer=L2(), activation='relu')(c_conv_3)
        c_conv_4 = BatchNormalization()(c_conv_4)
        c_conv_4 = MaxPooling2D()(c_conv_4)
        # 4 Layer
        c_conv_5 = Conv2D(32, (2, 2), kernel_regularizer=L2(), activation='relu')(c_conv_4)
        c_conv_5 = BatchNormalization()(c_conv_5)
        c_conv_5 = MaxPooling2D()(c_conv_5)

        c_cov_final = Flatten()(c_conv_5)

        # Dense layers
        c_dense_1 = Dense(64, kernel_regularizer=L2(), activation="relu", name="c_layer_1")(c_cov_final)
        c_dense_1 = BatchNormalization()(c_dense_1)
        # c_dense_1 = Dropout(0.5)(c_dense_1)
        c_dense_2 = Dense(128, kernel_regularizer=L2(), activation="relu", name="c_layer_2")(c_dense_1)
        c_dense_2 = BatchNormalization()(c_dense_2)
        # c_dense_2 = Dropout(0.5)(c_dense_2)
        c_dense_3 = Dense(64, kernel_regularizer=L2(), activation="relu", name="c_layer_3")(c_dense_2)
        c_dense_3 = BatchNormalization()(c_dense_3)
        # c_dense_3 = Dropout(0.5)(c_dense_3)
        c_dense_4 = Dense(32, kernel_regularizer=L2(), activation="relu", name="c_final_layer")(c_dense_3)
        c_dense_4 = BatchNormalization()(c_dense_4)
        # c_dense_4 = Dropout(0.5)(c_dense_4)

        # Output
        output_layer_c = Dense(1, activation='sigmoid', name="output_layer_c")(c_dense_4)

        model_c = Model(inputs=[c_ip_img], outputs=[output_layer_c], name="model_c")
        self.model = model_c
