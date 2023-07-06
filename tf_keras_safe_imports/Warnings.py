import warnings


class Warnings:
    @staticmethod
    def tf_import_warning():
        warnings.warn(
            "You are likely using an IDE, tensorflow.keras is lazy loaded, meaning it only holds a reference until use,"
            " therefore IDEs only know about the reference tensorflow holds to the keras module and not its content. "
            "To work around this issue alter the __init__.py file of tensorflow, or ignore to continue using the "
            "keras.api._v2 protected module, which is the same loaded by tensorflow in __init__.py as of the version"
            " this code was written for(2.12), more info on: https://github.com/tensorflow/tensorflow/issues/53144"
        )
