from tf_keras_safe_imports.Warnings import Warnings

# Assumes that if an import error happens its due to the issue detailed on the print below and a workaround is used.
try:
    from tensorflow.keras.models import Model
except ImportError:
    from keras.api._v2.keras.models import Model
    Warnings.tf_import_warning()

# Export the necessary modules so they can be imported elsewhere
__all__ = [
    'Model',
]
