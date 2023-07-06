import shutil
from ImageProcessor import ImageProcessor
from PIL import Image
import os
import cv2 as cv


# Class to facilitate manipulating files, used to build the combined dataset
class FileManipulator:
    def __init__(self):
        pass

    # Checks if a given filename contains any string from a list of restrictions.
    # It has an optional argument to ignore case.
    @staticmethod
    def has_any_restrictions(name, restrictions, ignore_case=True):
        if ignore_case:
            restrictions = [restriction.lower() for restriction in restrictions]
            name = name.lower()
        return any(restriction in name for restriction in restrictions)

    # checks if a file name contains at least one string from a list of inclusions and none from a list of exclusions
    # for each segment of the file path (i.e., root directory, file name, and file extension)
    # It is used to apply restrictions to the filename, the file path or the file type.
    @staticmethod
    def has_any_of_multiple_restrictions(segments, inclusions, exclusions):
        for segment, inclusion_list, exclusion_list in zip(segments, inclusions, exclusions):
            if inclusion_list != [] and not FileManipulator.has_any_restrictions(segment, inclusion_list):
                return True
            if FileManipulator.has_any_restrictions(segment, exclusion_list):
                return True
        return False

    # processes all JPEG images in the specified directory using error level analysis
    @staticmethod
    def batch_method(origin_directory, target_directory, method):
        # Check if the target directory exists, and create it if not
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        # Walk through the origin directory and process all JPEG images
        for root, dirs, files in os.walk(origin_directory):
            for file in files:
                print(f"Processing {file} -> {root}")
                if os.path.splitext(file)[1] != '.jpg':
                    print(f"Skipping {file}: not a JPEG image")
                    continue
                img = cv.imread(os.path.join(root, file))
                img = method(img)

                img.save(os.path.join(target_directory, file), 'JPEG', quality=100)

    # processes all JPEG images in the specified directory using error level analysis
    @staticmethod
    def batch_method_1_ela(origin_directory, target_directory, quality=95):
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)
        for root, dirs, files in os.walk(origin_directory):
            for file in files:
                print(f"filename:{file} root:{root}")
                if os.path.splitext(file)[1] != '.jpg':
                    print("Invalid type")
                    break
                img = cv.imread(os.path.join(root, file))
                img = ImageProcessor.method_1_ela(img, quality)
                img.save(os.path.join(target_directory, file), 'JPEG', quality=100)

    # Processes all JPEG images in the specified directory using the method 2 based on dwt
    @staticmethod
    def batch_method_2_dwt(origin_directory, target_directory):
        # Check if the target directory exists, and create it if not
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        # Walk through the origin directory and process all JPEG images
        for root, dirs, files in os.walk(origin_directory):
            for file in files:
                print(f"Processing {file} -> {root}")
                if os.path.splitext(file)[1] != '.jpg':
                    print(f"Skipping {file}: not a JPEG image")
                    continue
                img = cv.imread(os.path.join(root, file))
                img = ImageProcessor.method_2_dwt(img)

                img.save(os.path.join(target_directory, file), 'JPEG', quality=100)

    # renames files in origin_directory by adding a prefix to them, but only if the file paths do not contain any of the
    # specified exclusions and do contain at least one of the specified inclusions
    # prefix: word to be added before filename
    # keep_old_name: bool to keep old filename or name IMG_Number
    # exclusions: won't rename if on root, , example root restrictions: '_orig' and '_mask'
    @staticmethod
    def batch_add_prefix(origin_directory, keep_old_name=True, prefix='_', root_inclusions=[], root_exclusions=[],
                         name_inclusions=[], name_exclusions=[], type_inclusions=[], type_exclusions=[]):
        count = 0
        for root, dirs, files in os.walk(origin_directory):
            for file in files:
                print(f"N {count} Processing {file} -> {root}")
                name_type = os.path.splitext(file)
                root_name_type = [root, name_type[0], name_type[1][1:]]
                inclusions = [root_inclusions, name_inclusions, type_inclusions]
                exclusions = [root_exclusions, name_exclusions, type_exclusions]
                if FileManipulator.has_any_of_multiple_restrictions(root_name_type, inclusions, exclusions):
                    continue
                print('valid path')
                old_name = os.path.join(root, file)
                new_name = os.path.join(root, prefix + file) if keep_old_name else \
                    os.path.join(root, prefix + "IMG_" + str(count) + name_type[1].lower())
                os.rename(old_name, new_name)
                count += 1

    # copies all files (including subfolders) in the specified origin_directory to the target_directory, but only if the
    # file paths do not contain any of the specified exclusions and do contain at least one of the specified inclusions
    @staticmethod
    def batch_copy(origin_directory, target_directory, root_inclusions=[], root_exclusions=[], name_inclusions=[],
                   name_exclusions=[], type_inclusions=[], type_exclusions=[]):
        for root, dirs, files in os.walk(origin_directory):
            for file in files:
                print(f"filename:{file} root:{root}")
                origin_file_path = os.path.join(root, file)
                target_file_path = os.path.join(target_directory, file)
                name_type = os.path.splitext(file)
                root_name_type = [root, name_type[0], name_type[1][1:]]
                inclusions = [root_inclusions, name_inclusions, type_inclusions]
                exclusions = [root_exclusions, name_exclusions, type_exclusions]
                if FileManipulator.has_any_of_multiple_restrictions(root_name_type, inclusions, exclusions):
                    continue
                print('valid path')
                shutil.copy2(origin_file_path, target_file_path)

    # converts all files in the origin_directory to JPEG format, but only if the file type do not contain any of the
    # specified exclusions and do contain at least one of the specified inclusions
    @staticmethod
    def convert_all_to_jpg(origin_directory, type_inclusions=[], type_exclusions=[]):
        if "jpg" not in type_exclusions:
            type_exclusions.append("jpg")
        for root, dirs, files in os.walk(origin_directory):
            for file in files:
                print(f"filename:{file} root:{root}")
                name_type = os.path.splitext(file)
                if FileManipulator.has_any_of_multiple_restrictions([name_type[1][1:]],
                                                                    [type_inclusions], [type_exclusions]):
                    continue
                outfile = os.path.join(root, name_type[0]) + ".jpg"
                try:
                    im = Image.open(os.path.join(root, file))
                    print(f"Generating jpeg for {file}")
                    im = im.convert('RGB')
                    im.thumbnail(im.size)
                    im.save(outfile, "JPEG", quality=100)
                    os.remove(os.path.join(root, file))
                except Exception as e:
                    print(e)
