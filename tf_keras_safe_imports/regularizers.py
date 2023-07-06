from tf_keras_safe_imports.Warnings import Warnings

# Assumes that if an import error happens its due to the issue detailed on the print below and a workaround is used.
try:
    from tensorflow.keras.regularizers import L2, L1L2
except ImportError:
    from keras.api._v2.keras.regularizers import L2, L1L2
    Warnings.tf_import_warning()

# Export the necessary modules so they can be imported elsewhere
__all__ = [
    'L2',
    'L1L2',
]
