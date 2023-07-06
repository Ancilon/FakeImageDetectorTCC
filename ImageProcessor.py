from io import BytesIO
from PIL import ImageChops, ImageEnhance, Image, ImageDraw
import cv2 as cv
import pywt
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import view_as_windows


# methods to perform feature selection that make manipulations easier to see on images, takes numpy.ndarray images
# as input and returns PIL.Image.Image as output
class ImageProcessor:
    def __init__(self):
        pass

    # Static method that performs Error Level Analysis (ELA) on an image using JPEG compression.
    # Returns a normalized difference image between the original image and a JPEG-compressed version of the image.
    @staticmethod
    def method_1_ela(image, quality=95):
        # Convert image to JPEG format with the specified quality
        temp_image = BytesIO()
        image = Image.fromarray(image)
        image.save(temp_image, 'JPEG', quality=quality)
        temp_image = Image.open(temp_image)

        # Calculate the difference between the original and the JPEG-compressed image
        difference_image = ImageChops.difference(image, temp_image)

        # Normalize the difference image
        extrema = difference_image.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1
        scale = 255.0 / max_diff
        normalized_image = ImageEnhance.Brightness(difference_image).enhance(scale)

        return normalized_image

    # Static method that performs Discrete Wavelet Transform (DWT) on an image to highlight its edges.
    # Returns a PIL image that has been smoothed with a bilateral filter and had Laplacian edge detection applied.
    @staticmethod
    def method_2_dwt(image):
        # Convert image to grayscale and perform discrete wavelet transform
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        coeffs = pywt.dwt2(gray_image, 'bior1.3')
        LL, (LH, HL, HH) = coeffs

        # Reconstruct the image using only the high-frequency components
        high_freq_components = (None, (LH, HL, HH))
        joinedLhHlHh = pywt.idwt2(high_freq_components, 'db3')
        joinedLhHlHh = np.float32(joinedLhHlHh)

        # Apply bilateral filter to smooth the image while preserving edges
        blurred = cv.bilateralFilter(joinedLhHlHh, 9, 75, 75)
        blurred = np.uint8(blurred)

        # Apply Laplacian edge detection to highlight edges
        kernel_size = 3
        imgLapacian = cv.Laplacian(blurred, cv.CV_16S, ksize=kernel_size)
        abs_dst = cv.convertScaleAbs(imgLapacian)

        # Convert result back to PIL image format
        final_image = Image.fromarray(abs_dst)

        return final_image

    # this method highlights high contrast areas of the image similar to the method 2, was not used for the final work
    @staticmethod
    def method_3(image):
        # Convert the image to grayscale for easier manipulation
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # Use the Canny edge detector to find areas of the image with high contrast
        edges = cv.Canny(gray_image, 100, 200)

        # Dilate the image to make the edges more pronounced
        edges = cv.dilate(edges, None)

        # Use contour detection to find areas of the image with sharp changes in color
        contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Iterate through the contours and draw rectangles around areas with high contrast
        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)
            cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Convert the modified image back to PIL format
        pil_image = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))

        return pil_image

    @staticmethod
    def method_4(image, threshold=0.9):
        # Convert the image to grayscale
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # Apply a threshold to create a binary image
        _, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)

        # Find contours in the binary image
        contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Compute the similarity between pairs of contours
        num_contours = len(contours)
        similarities = np.zeros((num_contours, num_contours))
        for i in range(num_contours):
            for j in range(i + 1, num_contours):
                similarities[i, j] = cv.matchShapes(contours[i], contours[j], cv.CONTOURS_MATCH_I2, 0)

        # Draw lines between similar contours
        height, width, _ = image.shape
        for i in range(num_contours):
            for j in range(i + 1, num_contours):
                similarity = similarities[i, j]
                if similarity < threshold:
                    # Compute the centroid of each contour
                    centroid_i = np.mean(contours[i], axis=0, dtype=np.int32)
                    centroid_j = np.mean(contours[j], axis=0, dtype=np.int32)

        # Convert the image back to PIL Image format
        pil_image = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))

        return pil_image

    @staticmethod
    def method_5(image):
        # Convert the image to grayscale
        grayscale_image = Image.fromarray(np.uint8(image)).convert('L')

        # Calculate the local variance or noise level estimate
        local_variance = np.var(image, axis=(0, 1))

        # Normalize the local variance values between 0 and 255
        normalized_variance = (local_variance - np.min(local_variance)) / (
                    np.max(local_variance) - np.min(local_variance))
        normalized_variance = np.uint8(normalized_variance * 255)

        # Create a PIL Image from the normalized variance values
        noise_map = Image.fromarray(normalized_variance)

        # Resize the noise map to match the dimensions of the RGB image
        noise_map_resized = noise_map.resize(grayscale_image.size)

        # Convert the noise map to RGB mode
        noise_map_rgb = noise_map_resized.convert('RGB')

        # Apply a color mapping or gradient to visualize the noise map
        colormap = np.stack((255 - normalized_variance, normalized_variance, np.zeros_like(normalized_variance)),
                            axis=-1)
        noise_map_colored = Image.fromarray(colormap)

        # Convert the grayscale image to RGB
        rgb_image = grayscale_image.convert('RGB')

        # Overlay the noise map on the RGB image
        highlighted_image = Image.blend(rgb_image, noise_map_rgb, alpha=0.5)

        return highlighted_image

    @staticmethod
    def method_6(image, patch_size=3, overlap=1):
        # Convert the image to grayscale
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # Divide the image into overlapping patches
        window_shape = (patch_size, patch_size)
        patches = view_as_windows(image, window_shape, step=overlap)

        # Calculate the local variance or noise level estimate for each patch
        noise_levels = np.zeros(patches.shape[:2])
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                patch = patches[i, j]
                noise_levels[i, j] = np.var(patch)

        # Construct the noise map by creating a new image where the intensity or color of each pixel
        # represents the noise level estimate
        noise_map = np.zeros(image.shape)
        for i in range(noise_levels.shape[0]):
            for j in range(noise_levels.shape[1]):
                x_start = i * overlap
                x_end = x_start + patch_size
                y_start = j * overlap
                y_end = y_start + patch_size
                noise_map[x_start:x_end, y_start:y_end] = noise_levels[i, j]

        # Normalize the noise map to the range [0, 255] and convert it to 8-bit integer values
        noise_map = (noise_map - noise_map.min()) / (noise_map.max() - noise_map.min()) * 255
        noise_map = noise_map.astype(np.uint8)

        # Convert the resulting noise map to a PIL Image and return it
        return Image.fromarray(noise_map)





