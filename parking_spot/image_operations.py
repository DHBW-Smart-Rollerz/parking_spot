import cv2
import numpy as np

max_white_ratio = 0.15 # max white value of parking spot in binary


def DetFilter(image):
    blurred_image = cv2.medianBlur(image, 1)
    ret, binary_image = cv2.threshold(blurred_image, 85, 255, cv2.THRESH_BINARY)
    return binary_image

def getBestMatchingSpots(spots, spots_w, image):

    ratios = []
    for spot in spots:
        points = np.array([pt[0] for pt in spot[:4]], dtype=np.int32)
        sorted_points = points[points[:, 0].argsort()]




        if (sorted_points[0][0] < 0):
            sorted_points[0][0] = 0

        if (sorted_points[1][0] < 0):
            sorted_points[1][0] = 0

        binary_img = image

        # empty mask
        mask = np.zeros_like(binary_img, dtype=np.uint8)

        # fill form in mask
        cv2.fillPoly(mask, [points], 255)

        # extract area in mask
        masked = cv2.bitwise_and(binary_img, mask)

        # get white ratio
        total_area = cv2.countNonZero(mask)  
        white_pixels = cv2.countNonZero(masked)  
        white_ratio = white_pixels / total_area if total_area > 0 else 0
        ratios.append(white_ratio)
    
    ratios = np.array(ratios)
    # index of the minimum ratio
    if (ratios.size == 0):
        return None, None
    min_idx = np.argmin(ratios)
    
    if (ratios[min_idx] < max_white_ratio):
        return spots[min_idx], spots_w[min_idx]
    else:
        return None
        
    