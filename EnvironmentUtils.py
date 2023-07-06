import os
import tensorflow as tf


# A utility class for managing hardware resources.
class EnvironmentUtils:
    def __init__(self):
        pass

    @staticmethod
    # Prints information about the available GPUs
    def show_gpu_info():
        gpus = tf.config.list_physical_devices('GPU')
        print(len(gpus), "Physical GPUs,")

    @staticmethod
    # Limit memory used by gpu, example 1024 equivalent of 1GB.
    # Raises RunTimeError if virtual devices are set after GPUs have been initialized.
    def limit_gpu_memory(memory_limit):
        # GPU configuration
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)])
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e)

    @staticmethod
    # Disables the GPU, useful for testing CPU training speed if cuda is set up.
    def disable_gpu():
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    @staticmethod
    # Limit the amount of GPU memory growth by setting the allow_growth option for TensorFlow to dynamically allocate
    # memory as needed.
    def set_memory_growth(is_enabled):
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        if gpu_devices:
            for device in gpu_devices:
                tf.config.experimental.set_memory_growth(device, is_enabled)

    @staticmethod
    # sets the logging level of tensorflow in 2 different ways
    # '0' Default behavior. Shows all log messages, including debug, info, warning, and error messages.
    # '1' Filters out INFO log messages and displays only warning and error messages.
    # '2' Filters out INFO and WARNING log messages and displays only error messages.
    # '3' Filters out all log messages except for error messages.
    # ----------------------------------------------------------------------------------------
    # 'FATAL': Indicates a critical error that causes TensorFlow to terminate immediately.
    # 'ERROR': Represents an error condition that may impact the execution of the program.
    # WARNING: Indicates a potential issue or a non-fatal error.
    # INFO: Provides informational messages about the progress or status of the program.
    # DEBUG: Displays detailed debugging information, which can be useful for troubleshooting.
    def set_tensorflow_log_level(level_os='0', level_tf='INFO'):
        valid_os_levels = ['0', '1', '2', '3']
        assert level_os in valid_os_levels, f"Invalid TensorFlow log level. Please choose from: {', '.join(valid_os_levels)}"

        valid_tf_levels = ['FATAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG']
        assert level_tf in valid_tf_levels, f"Invalid TensorFlow log level. Please choose from: {', '.join(valid_tf_levels)}"

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = level_os
        tf.get_logger().setLevel(level_tf)

    @staticmethod
    # set tf random seed for consistent random values
    def set_tf_seed(seed):
        tf.random.set_seed(seed)
