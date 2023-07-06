from ImageProcessor import ImageProcessor
import os
from FileManipulator import FileManipulator
import cv2 as cv


def print_termination(message):
    # Use a breakpoint in the code line below to debug your script.
    print(f'{message}')  # Press Ctrl+F8 to toggle the breakpoint.


if __name__ == '__main__':
    '''
    img = cv.imread('D:/TCC/Assets/test2.jpg')
    cv.imshow('Original image', img)
    #cv.waitKey(0)
    ela_image = ImageProcessor.method_1_ela(img, 95)
    #cv.imshow('ELA image', ela_image)
    ela_image.show()
    print_termination("program terminated")
    '''
    img_path = 'D:/TCC/TrabRevistaMarcelo/duplicated1.jpg'

    img = cv.imread(img_path)
    ela_img = ImageProcessor.method_1_ela(img)
    ela_img.save(('D:/TCC/TrabRevistaMarcelo/testImages/ManipulatedELA.jpg'), 'JPEG', quality=100)
    dwt_img = ImageProcessor.method_2_dwt(img)
    dwt_img.save(('D:/TCC/TrabRevistaMarcelo/testImages/ManipulatedDWT.jpg'), 'JPEG', quality=100)
    # chatgpt_img = ImageProcessor.method_3(img)
    # chatgpt_img.save(('D:/TCC/TrabRevistaMarcelo/testImages/ManipulatedMethod3.jpg'), 'JPEG', quality=100)
    chatgpt2_img = ImageProcessor.method_4(img)
    chatgpt2_img.save(('D:/TCC/TrabRevistaMarcelo/testImages/ManipulatedMethod4.jpg'), 'JPEG', quality=100)
    chatgpt3 = ImageProcessor.method_5(img)
    chatgpt3.save(('D:/TCC/TrabRevistaMarcelo/testImages/ManipulatedMethod5.jpg'), 'JPEG', quality=100)
    method6 = ImageProcessor.method_6(img)
    method6.save(('D:/TCC/TrabRevistaMarcelo/testImages/ManipulatedMethod6.jpg'), 'JPEG', quality=100)

    origin_directory = 'D:/TCC/Assets/Datasets/temp2'
    target_directory = 'D:/TCC/Assets/Datasets/temp'
    # FileManipulator.batch_add_prefix(target_directory, keep_old_name=False, prefix='more_yay_', type_exclusions=['bmp'])
    # FileManipulator.batch_copy(origin_directory, target_directory, root_inclusions=['tampered-realistic'])
    # FileManipulator.convert_all_to_jpg(target_directory)

    FileManipulator.batch_method('D:/TCC/Assets/Datasets/CASIA_v2.0/normal/Original',
                                 'D:/TCC/Assets/Datasets/CASIA_v2.0/noise_analysis/Original',
                                 ImageProcessor.method_6)
    FileManipulator.batch_method('D:/TCC/Assets/Datasets/CASIA_v2.0/normal/Tampered',
                                 'D:/TCC/Assets/Datasets/CASIA_v2.0/noise_analysis/Tampered',
                                 ImageProcessor.method_6)

    # FileManipulator.convert_all_to_jpg('D:/TCC/Assets/Datasets/FinalDataset/normal/Original')
    # FileManipulator.convert_all_to_jpg('D:/TCC/Assets/Datasets/FinalDataset/normal/Tampered')

    # FileManipulator.batch_method_1_ela('D:/TCC/Assets/Datasets/FinalDataset/normal/Original',
    #   'D:/TCC/Assets/Datasets/FinalDataset/method_1_ela/Original')
    # FileManipulator.batch_method_1_ela('D:/TCC/Assets/Datasets/FinalDataset/normal/Tampered',
    # 'D:/TCC/Assets/Datasets/FinalDataset/method_1_ela/Tampered')

    # FileManipulator.batch_method_2_dwt('D:/TCC/Assets/Datasets/CASIA_v2.0/normal/Original',
    #                                           'D:/TCC/Assets/Datasets/CASIA_v2.0/dwt_bilateral/Original')
    # FileManipulator.batch_method_2_dwt('D:/TCC/Assets/Datasets/CASIA_v2.0/normal/Tampered',
    #                                           'D:/TCC/Assets/Datasets/CASIA_v2.0/dwt_bilateral/Tampered')
