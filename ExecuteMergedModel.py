from ModelA import ModelA
from ModelB import ModelB
from ModelC import ModelC
from MergedModel import MergedModel
from ModelTrainingUtils import ModelTrainingUtils
from EnvironmentUtils import EnvironmentUtils

# IDE safe imports
from tf_keras_safe_imports.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, TerminateOnNaN
from tf_keras_safe_imports.utils import plot_model


def create_callbacks(model_name):
    csv_logger = CSVLogger(f'TrainingLogs/{model_name}.log')
    terminate_on_nan = TerminateOnNaN()
    cp_callback = ModelCheckpoint(filepath=f'TempWeights/{model_name}.h5',
                                  save_best_only=True,
                                  save_weights_only=True,
                                  monitor='val_loss',
                                  mode='min',
                                  verbose=1
                                  )
    es_callback = EarlyStopping(
        monitor='val_loss',
        patience=100,
        verbose=1,
        restore_best_weights='True'
    )
    return [csv_logger, terminate_on_nan, cp_callback, es_callback]


if __name__ == '__main__':
    _normal_dataset = 'D:/TCC/Assets/Datasets/CASIA_v2.0/normal'
    _ela_dataset = 'D:/TCC/Assets/Datasets/CASIA_v2.0/error_level_analysis'
    _dwt_bilateral_dataset = 'D:/TCC/Assets/Datasets/CASIA_v2.0/dwt_bilateral'

    EnvironmentUtils.show_gpu_info()

    callbacks = create_callbacks("ModelA")
    _Model_A = ModelA(_normal_dataset, callbacks)
    _Model_A.compile_model()
    _Model_A.fit_model()

    callbacks = create_callbacks("ModelB")
    _Model_B = ModelB(_ela_dataset, callbacks)
    _Model_B.compile_model()
    _Model_B.fit_model()

    callbacks = create_callbacks("ModelC")
    _Model_C = ModelC(_dwt_bilateral_dataset, callbacks)
    _Model_C.compile_model()
    _Model_C.fit_model()

    ModelTrainingUtils.freeze_model(_Model_A.model, 0)
    ModelTrainingUtils.freeze_model(_Model_B.model, 0)
    ModelTrainingUtils.freeze_model(_Model_C.model, 0)

    callbacks = create_callbacks("MergedModel")
    _Merged_Model = MergedModel(_Model_A, _Model_B, _Model_C, callbacks)
    _Merged_Model.compile_model()
    _Merged_Model.fit_model()

    # plot model architecture
    plot_model(_Model_A.model, "Architecture/model_a_architecture.png", show_shapes=True)
    plot_model(_Model_B.model, "Architecture/model_b_architecture.png", show_shapes=True)
    plot_model(_Model_C.model, "Architecture/model_c_architecture.png", show_shapes=True)
    plot_model(_Merged_Model.model, "Architecture/merged_architecture.png", show_shapes=True)

    # plot model training data
    save_folder_name = 'TrainingData'
    ModelTrainingUtils.plot_model_training_data(_Model_A.history, 'A_Model', 'accuracy', save_folder_name)
    ModelTrainingUtils.plot_model_training_data(_Model_A.history, 'A_Model', 'loss', save_folder_name)
    ModelTrainingUtils.plot_model_roc(_Model_A.model, _Model_A.test_dataset, 'A_Model', save_folder_name)

    ModelTrainingUtils.plot_model_training_data(_Model_B.history, 'B_Model', 'accuracy', save_folder_name)
    ModelTrainingUtils.plot_model_training_data(_Model_B.history, 'B_Model', 'loss', save_folder_name)
    ModelTrainingUtils.plot_model_roc(_Model_B.model, _Model_B.test_dataset, 'B_Model', save_folder_name)

    ModelTrainingUtils.plot_model_training_data(_Model_C.history, 'C_Model', 'accuracy', save_folder_name)
    ModelTrainingUtils.plot_model_training_data(_Model_C.history, 'C_Model', 'loss', save_folder_name)
    ModelTrainingUtils.plot_model_roc(_Model_C.model, _Model_C.test_dataset, 'C_Model', save_folder_name)

    ModelTrainingUtils.plot_model_training_data(_Merged_Model.history, 'Merged_Model', 'accuracy', save_folder_name)
    ModelTrainingUtils.plot_model_training_data(_Merged_Model.history, 'Merged_Model', 'loss', save_folder_name)
    ModelTrainingUtils.plot_model_roc(_Merged_Model.model, _Merged_Model.test_dataset, 'Merged_Model', save_folder_name)

    # saving models
    _Model_A.model.save('SavedModels/ModelA/modelA', save_format='h5')
    _Model_B.model.save('SavedModels/ModelB/modelB', save_format='h5')
    _Model_C.model.save('SavedModels/ModelC/modelC', save_format='h5')
    _Merged_Model.model.save('SavedModels/ModelMerged/modelMerged', save_format='h5')
