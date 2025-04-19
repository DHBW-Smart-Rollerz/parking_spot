import cv2


def DetFilter(image):
    blurred_image = cv2.medianBlur(image, 1)
    ret, binary_image = cv2.threshold(blurred_image, 85, 255, cv2.THRESH_BINARY)
    return binary_image