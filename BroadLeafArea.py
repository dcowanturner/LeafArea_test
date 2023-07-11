import cv2
import numpy as np
import os
import pytesseract
import pandas as pd

def process_image(image_path, output_image_folder, filename, min_leaf_size=500, threshold_value=128):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color range for the ruler
    lower_color = np.array([40, 50, 50])
    upper_color = np.array([160, 255, 255])

    # Create a mask using the color range
    mask_ruler = cv2.inRange(hsv, lower_color, upper_color)

    # Apply a closing operation to the ruler mask
    kernel = np.ones((5, 5), np.uint8)
    mask_ruler = cv2.morphologyEx(mask_ruler, cv2.MORPH_CLOSE, kernel)

    # Apply a dilation operation to the ruler mask
    dilated_mask_ruler = cv2.dilate(mask_ruler, kernel, iterations=1)

    # Find contours of the ruler
    cropped_mask = dilated_mask_ruler[int(0.75 * mask_ruler.shape[0]):, :]
    contours_ruler, _ = cv2.findContours(cropped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Offset y-coordinates of the contours
    offset = int(0.75 * mask_ruler.shape[0])
    contours_ruler = [contour + np.array([0, offset]) for contour in contours_ruler]

    # Calculate the area of the ruler and the conversion factor
    real_ruler_area = 15.84 * 2.94  # width*height in cm^2
    conversion_factor = 0
    min_y = image.shape[0]
    for contour in contours_ruler:
        if cv2.contourArea(contour) > 500:  # filter out very small contours
            # Get the rotated rectangle that encloses the contour
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int64(box)

            # Draw the rotated rectangle on the original image
            cv2.polylines(image, [box], True, (0, 255, 0), 2)  # green contour lines

            # Calculate the area of the rectangle in pixels
            pixel_area = rect[1][0] * rect[1][1]

            conversion_factor = real_ruler_area / pixel_area
            # Update minimum y-coordinate
            min_y = min(min_y, np.min(box[:, 1]))
    
    hsv = hsv[:min_y, :]

    cols = hsv.shape[1]
    left_margin = int(0.1 * cols)
    right_margin = int(0.9 * cols)
    hsv = hsv[:, left_margin:right_margin]

    height, width = hsv.shape[:2]
    crop_height = int(height * 0.3)
    cropped_image = hsv[crop_height:, :]

    _, thresholded = cv2.threshold(cropped_image, threshold_value, 255, cv2.THRESH_BINARY)

    merged_image = thresholded.copy()
    merged_image[:, :, 1] = np.where((thresholded[:, :, 1] > 0) | (thresholded[:, :, 2] == 0), 255, merged_image[:, :, 1])

    grayscale = cv2.cvtColor(merged_image, cv2.COLOR_BGR2GRAY)

    leaf_pixels = grayscale == 150

    filtered_image = np.zeros_like(grayscale)

    filtered_image[leaf_pixels] = 150

    kernel_close = np.ones((9, 9), np.uint8)
    closed_image = cv2.morphologyEx(filtered_image, cv2.MORPH_CLOSE, kernel_close)

    ret, binary_image = cv2.threshold(closed_image, 127, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    large_objects = []
    min_object_area = 10000  # Minimum object area threshold

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_object_area:
            large_objects.append(contour)

    offset_x = left_margin
    offset_y = crop_height

    large_objects = [
        contour + np.array([[offset_x, offset_y]])
        for contour in large_objects
    ]

    result_image = image.copy()
    cv2.drawContours(result_image, large_objects, -1, (0, 255, 0), thickness=2)

    cv2.imwrite(os.path.join(output_image_folder, filename), result_image)

    total_leaf_area = np.sum(closed_image == 150)

    total_leaf_area_cm2 = total_leaf_area * conversion_factor

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    text = pytesseract.image_to_string(gray)

    return total_leaf_area_cm2, text

def main(input_folder_path, output_image_folder):
    os.makedirs(output_image_folder, exist_ok=True)

    df = pd.DataFrame(columns=['Filename', 'Leaf Area', 'Text'])

    for filename in os.listdir(input_folder_path):
        if not (filename.endswith('.jpg') or filename.endswith('.png')):
            continue
        
        image_path = os.path.join(input_folder_path, filename)

        leaf_area, text = process_image(image_path, output_image_folder, filename)
        new_row = {'Filename': filename, 'Leaf Area': leaf_area, 'Text': text}
        df = pd.concat([df, pd.DataFrame(new_row, index=[0])], ignore_index=True)

    df.to_csv('output.csv', index=False)

if __name__ == "__main__":
    main('./Test', './output_images')
