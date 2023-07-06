from ModelA import ModelA
from ModelB import ModelB
from ModelC import ModelC
from MergedModel import MergedModel
from ModelTrainingUtils import ModelTrainingUtils
from EnvironmentUtils import EnvironmentUtils
from DatasetAugmentationCallback import DatasetAugmentationCallback

# IDE safe imports
from tf_keras_safe_imports.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, TerminateOnNaN


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
        patience=50,
        verbose=1,
        restore_best_weights='True'
    )
    return [csv_logger, terminate_on_nan, cp_callback, es_callback]


if __name__ == '__main__':
    normal_dataset = 'D:/TCC/Assets/Datasets/CASIA_v2.0/normal'
    ela_dataset = 'D:/TCC/Assets/Datasets/CASIA_v2.0/error_level_analysis'
    dwt_bilateral_dataset = 'D:/TCC/Assets/Datasets/CASIA_v2.0/dwt_bilateral'

    #EnvironmentUtils.disable_gpu()
    EnvironmentUtils.show_gpu_info()
    EnvironmentUtils.set_memory_growth(True)


    callbacks = create_callbacks("ModelA")
    Model_A = ModelA(normal_dataset, callbacks)
    Model_A.add_callback(DatasetAugmentationCallback(Model_A))
    Model_A.compile_model()
    Model_A.fit_model()
    #Model_A.load_model_weights('D:/ancil/PycharmProjects/FakeImageDetector/SavedModels/ModelA/modelA.h5')

    callbacks = create_callbacks("ModelB")
    Model_B = ModelB(ela_dataset, callbacks)
    Model_B.add_callback(DatasetAugmentationCallback(Model_B))
    Model_B.compile_model()
    Model_B.fit_model()
    #Model_B.load_model_weights('D:/ancil/PycharmProjects/FakeImageDetector/SavedModels/ModelB/modelB.h5')

    callbacks = create_callbacks("ModelC")
    Model_C = ModelC(dwt_bilateral_dataset, callbacks)
    Model_C.add_callback(DatasetAugmentationCallback(Model_C))
    Model_C.compile_model()
    Model_C.fit_model()
    #Model_C.load_model_weights('D:/ancil/PycharmProjects/FakeImageDetector/SavedModels/ModelC/modelC.h5')

    ModelTrainingUtils.freeze_model(Model_A.model, 0)
    ModelTrainingUtils.freeze_model(Model_B.model, 0)
    ModelTrainingUtils.freeze_model(Model_C.model, 0)

    callbacks = create_callbacks("MergedModel")
    Merged_Model = MergedModel(Model_A, Model_B, Model_C, callbacks)
    # Merged_Model.add_callback(DatasetAugmentationCallback(Merged_Model))
    Merged_Model.compile_model()
    Merged_Model.fit_model()
    #Merged_Model.load_model_weights('D:/ancil/PycharmProjects/FakeImageDetector/SavedModels/ModelMerged/modelMerged.h5')

    # evaluate models
    Model_A.reload_datasets()
    Model_B.reload_datasets()
    Model_C.reload_datasets()
    Merged_Model.reload_datasets()

    Model_A.evaluate_model_training()
    Model_A.evaluate_model_validation()
    Model_A.evaluate_model_test()

    Model_B.evaluate_model_training()
    Model_B.evaluate_model_validation()
    Model_B.evaluate_model_test()

    Model_C.evaluate_model_training()
    Model_C.evaluate_model_validation()
    Model_C.evaluate_model_test()

    Merged_Model.evaluate_model_training()
    Merged_Model.evaluate_model_validation()
    Merged_Model.evaluate_model_test()

    # calculate mse
    ModelTrainingUtils.calculate_mse(Model_A.model, Model_A.validation_dataset, 'A_Model')
    ModelTrainingUtils.calculate_mse(Model_A.model, Model_A.test_dataset, 'A_Model')

    ModelTrainingUtils.calculate_mse(Model_B.model, Model_B.validation_dataset, 'B_Model')
    ModelTrainingUtils.calculate_mse(Model_B.model, Model_B.test_dataset, 'B_Model')

    ModelTrainingUtils.calculate_mse(Model_C.model, Model_C.validation_dataset, 'C_Model')
    ModelTrainingUtils.calculate_mse(Model_C.model, Model_C.test_dataset, 'C_Model')

    ModelTrainingUtils.calculate_mse(Merged_Model.model, Merged_Model.validation_dataset, 'Merged_Model')
    ModelTrainingUtils.calculate_mse(Merged_Model.model, Merged_Model.test_dataset, 'Merged_Model')

    # plot model ROC on validation and test datasets
    save_folder_name = 'TrainingData'
    ModelTrainingUtils.plot_model_roc(Model_A.model, Model_A.validation_dataset, 'A_Model_Validation_', save_folder_name)
    ModelTrainingUtils.plot_model_roc(Model_A.model, Model_A.test_dataset, 'A_Model_Test_', save_folder_name)

    ModelTrainingUtils.plot_model_roc(Model_B.model, Model_B.validation_dataset, 'B_Model_Validation_', save_folder_name)
    ModelTrainingUtils.plot_model_roc(Model_B.model, Model_B.test_dataset, 'B_Model_Test_', save_folder_name)

    ModelTrainingUtils.plot_model_roc(Model_C.model, Model_C.validation_dataset, 'C_Model_Validation_', save_folder_name)
    ModelTrainingUtils.plot_model_roc(Model_C.model, Model_C.test_dataset, 'C_Model_Test_', save_folder_name)

    ModelTrainingUtils.plot_model_roc(Merged_Model.model, Merged_Model.validation_dataset, 'Merged_Model_Validation_', save_folder_name)
    ModelTrainingUtils.plot_model_roc(Merged_Model.model, Merged_Model.test_dataset, 'Merged_Model_Test_', save_folder_name)

    # plot model architecture
    save_folder_name = 'Architecture'
    ModelTrainingUtils.plot_model_architecture_visualkeras(Model_A.model, 'A_Model', save_folder_name)
    ModelTrainingUtils.plot_model_architecture_tensorflow(Model_A.model, 'A_Model', save_folder_name)
    
    ModelTrainingUtils.plot_model_architecture_visualkeras(Model_B.model, 'B_Model', save_folder_name)
    ModelTrainingUtils.plot_model_architecture_tensorflow(Model_B.model, 'B_Model', save_folder_name)
    
    ModelTrainingUtils.plot_model_architecture_visualkeras(Model_C.model, 'C_Model', save_folder_name)
    ModelTrainingUtils.plot_model_architecture_tensorflow(Model_C.model, 'C_Model', save_folder_name)
    
    ModelTrainingUtils.plot_model_architecture_visualkeras(Merged_Model.model, 'Merged_Model', save_folder_name)
    ModelTrainingUtils.plot_model_architecture_tensorflow(Merged_Model.model, 'Merged_Model', save_folder_name)

    # plot model training data
    save_folder_name = 'TrainingData'
    ModelTrainingUtils.plot_model_training_data(Model_A.history, 'A_Model', 'accuracy', save_folder_name)
    ModelTrainingUtils.plot_model_training_data(Model_A.history, 'A_Model', 'loss', save_folder_name)

    ModelTrainingUtils.plot_model_training_data(Model_B.history, 'B_Model', 'accuracy', save_folder_name)
    ModelTrainingUtils.plot_model_training_data(Model_B.history, 'B_Model', 'loss', save_folder_name)

    ModelTrainingUtils.plot_model_training_data(Model_C.history, 'C_Model', 'accuracy', save_folder_name)
    ModelTrainingUtils.plot_model_training_data(Model_C.history, 'C_Model', 'loss', save_folder_name)

    ModelTrainingUtils.plot_model_training_data(Merged_Model.history, 'Merged_Model', 'accuracy', save_folder_name)
    ModelTrainingUtils.plot_model_training_data(Merged_Model.history, 'Merged_Model', 'loss', save_folder_name)

    # saving models
    Model_A.model.save('SavedModels/ModelA/modelA.h5', save_format='h5')
    Model_B.model.save('SavedModels/ModelB/modelB.h5', save_format='h5')
    Model_C.model.save('SavedModels/ModelC/modelC.h5', save_format='h5')
    Merged_Model.model.save('SavedModels/ModelMerged/modelMerged.h5', save_format='h5')
