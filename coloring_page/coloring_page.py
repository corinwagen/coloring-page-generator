from typing import Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image
import cv2

BoundaryExtractionMethod = Literal["character", "map", "adaptive_threshold", "color_boundaries"]
LineThickness = Literal["thin", "normal", "thick"]


def extract_boundaries(
    image_path: str,
    output_path: str | None = None,
    method: BoundaryExtractionMethod = "color_boundaries",
    line_thickness: LineThickness = "normal",
) -> Image:
    """
    Extract clean boundaries from colored maps using color segmentation.

    :param image_path: the input image path
    :param output_path: the output image path
    :param method: what algorithm to use
    :param line_thickness: how thick to make the lines
    :raises: ValueError for invalid method
    :returns: an image object for the coloring page
    """
    # Load image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    match method:
        case "character":
            result = extract_character_lines(img_rgb, line_thickness)
        case "color_boundaries" | "map":
            result = extract_color_boundaries(img_rgb, line_thickness)
        case "adaptive_threshold":
            result = extract_with_adaptive_threshold(img_rgb)
        case _:
            raise ValueError(f"unknown method `{method}`")

    # Convert to PIL Image
    result_pil = Image.fromarray(result)

    if output_path:
        result_pil.save(output_path)
        print(f"Coloring page saved to: {output_path}")

    return result_pil


def extract_character_lines(img_rgb: Image, line_thickness: LineThickness = "normal") -> NDArray[np.float64]:
    """
    Extract clean lines from character illustrations and artwork.

    :param img_rgb: the RGB input image
    :param line_thickness: how thick to make the lines
    :returns: a Numpy array of the coloring page
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Method 1: Adaptive threshold for main lines
    adaptive = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 6)

    # Method 2: Edge detection for details
    # Apply slight blur before edge detection to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Method 3: Laplacian for fine details
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    laplacian = np.absolute(laplacian)
    laplacian = np.uint8(np.clip(laplacian, 0, 255))
    _, laplacian = cv2.threshold(laplacian, 30, 255, cv2.THRESH_BINARY)

    # Combine methods
    combined = cv2.bitwise_and(adaptive, cv2.bitwise_not(edges))
    combined = cv2.bitwise_and(combined, cv2.bitwise_not(laplacian))

    # Clean up noise
    # Remove small components (noise)
    combined = remove_small_components_cv2(combined, min_area=25)

    # Fill small gaps
    kernel_close = np.ones((2, 2), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close)

    # Adjust line thickness
    thickness_settings = {"thin": ("erode", 1), "normal": (None, 0), "thick": ("dilate", 1)}

    operation, iterations = thickness_settings[line_thickness]
    if operation == "erode":
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        combined = cv2.erode(combined, kernel, iterations=iterations)
    elif operation == "dilate":
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        combined = cv2.dilate(combined, kernel, iterations=iterations)

    return combined


def extract_color_boundaries(img_rgb: Image, line_thickness: LineThickness = "normal") -> NDArray[np.float64]:
    """
    Extract clean lines from color boundaries.

    :param img_rgb: the RGB input image
    :param line_thickness: how thick to make the lines
    :returns: a Numpy array of the coloring page
    """
    # Adjust erosion based on desired line thickness
    erosion_iterations = {"thin": 1, "normal": 0, "thick": 0}[line_thickness]
    dilation_iterations = {"thin": 0, "normal": 0, "thick": 1}[line_thickness]

    # PATH 1: BOUNDARIES
    # Light bilateral filter for boundary detection (reduced from 9 to 5)
    blurred_boundaries = cv2.bilateralFilter(img_rgb, 5, 50, 50)

    # Convert to LAB color space (better for color segmentation)
    lab = cv2.cvtColor(blurred_boundaries, cv2.COLOR_RGB2LAB)

    # Find edges in each channel with adjusted thresholds
    edges_l = cv2.Canny(lab[:, :, 0], 40, 80)
    edges_a = cv2.Canny(lab[:, :, 1], 40, 80)
    edges_b = cv2.Canny(lab[:, :, 2], 40, 80)

    # Combine edges from all channels
    boundaries = cv2.bitwise_or(edges_l, cv2.bitwise_or(edges_a, edges_b))

    # Optional: thin the boundaries slightly
    kernel_thin = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    boundaries = cv2.morphologyEx(boundaries, cv2.MORPH_OPEN, kernel_thin, iterations=1)

    # PATH 2: TEXT AND FINE DETAILS
    # No blur for maximum sharpness on text
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Sharpen the image for better text detection
    kernel_sharpen = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    gray_sharp = cv2.filter2D(gray, -1, kernel_sharpen)

    # Higher threshold for crisp text detection
    edges_text = cv2.Canny(gray_sharp, 100, 250)

    # Adaptive threshold with smaller block size for finer detail
    text_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 3)
    text_adaptive = cv2.bitwise_not(text_adaptive)  # Invert to get text as white

    # Use morphological gradient for fine text details
    kernel_gradient = np.ones((2, 2), np.uint8)
    text_gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel_gradient)
    _, text_gradient = cv2.threshold(text_gradient, 30, 255, cv2.THRESH_BINARY)

    # Combine text detection methods
    text_features = cv2.bitwise_or(edges_text, text_adaptive)
    text_features = cv2.bitwise_or(text_features, text_gradient)

    # Remove only very tiny noise (reduced from 8 to 4)
    text_features = remove_small_components_cv2(text_features, min_area=4)

    all_edges = cv2.bitwise_or(boundaries, text_features)

    # Adjust line thickness
    kernel_thin = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    if erosion_iterations > 0:
        all_edges = cv2.erode(all_edges, kernel_thin, iterations=erosion_iterations)
    if dilation_iterations > 0:
        all_edges = cv2.dilate(all_edges, kernel_thin, iterations=dilation_iterations)

    # Invert (black lines on white background)
    result = cv2.bitwise_not(all_edges)

    return result


def extract_with_adaptive_threshold(img_rgb: Image) -> NDArray[np.float64]:
    """
    Use adaptive thresholding for maps with varying lighting.

    :param img_rgb: the RGB input image
    :param line_thickness: how thick to make the lines
    :returns: a Numpy array of the coloring page
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # adaptive threshold to handle varying lighting
    thresh1 = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # also get edges
    edges = cv2.Canny(enhanced, 30, 100)
    edges_inv = cv2.bitwise_not(edges)

    # combine threshold and edges
    combined = cv2.bitwise_and(thresh1, edges_inv)

    kernel = np.ones((2, 2), np.uint8)
    return cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)


def remove_small_components_cv2(binary_img: Image, min_area: float = 20) -> NDArray[np.float64]:
    """
    Remove small connected components.

    :param binary_img: the image
    :param min_area: the cutoff for what to remove
    :returns: a binary output image
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    output = np.zeros_like(binary_img)
    for label in range(1, num_labels):  # Skip background (label 0)
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            output[labels == label] = 255
    return output


