from tf_keras_safe_imports.Warnings import Warnings

# Assumes that if an import error happens its due to the issue detailed on the print below and a workaround is used.
try:
    from tensorflow.keras.layers import Input, Dense, concatenate, Flatten, Conv2D, MaxPooling2D,\
        BatchNormalization, Dropout
except ImportError:
    from keras.api._v2.keras.layers import Input, Dense, concatenate, Flatten, Conv2D, MaxPooling2D,\
        BatchNormalization, Dropout
    Warnings.tf_import_warning()

# Export the necessary modules so they can be imported elsewhere
__all__ = [
    'Input',
    'Dense',
    'concatenate',
    'Flatten',
    'Conv2D',
    'MaxPooling2D',
    'BatchNormalization',
    'Dropout',
]
