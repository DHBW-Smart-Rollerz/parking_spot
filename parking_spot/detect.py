import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from itertools import combinations

from parking_spot.corner_detection import CornerDetection
from camera_preprocessing.transformation.coordinate_transform import (
    CoordinateTransform,
)


class CornerDetector(Node):
    """A ROS2 node that subscribes to the undistorted image
    topic and processes the images."""

    def __init__(self):
        super().__init__("undistorted_image_subscriber")

        # Declare and get parameters
        self.declare_parameter("undistorted_image_topic", "/camera/image/undistorted")

        undistorted_image_topic = (
            self.get_parameter("undistorted_image_topic")
            .get_parameter_value()
            .string_value
        )

        # Create a subscription to the undistorted image topic
        self.subscription = self.create_subscription(
            Image,
            undistorted_image_topic,
            self.listener_callback,
            10,
        )
        self.subscription  # prevent unused variable warning

        # Initialize CvBridge
        self.bridge = CvBridge()

        self.get_logger().info(f"Subscribed to {undistorted_image_topic}")

    def listener_callback(self, msg):
        """Callback function for the undistorted image subscriber."""
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="8UC1")
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return

        # Process the image using OpenCV
        processed_image = self.process_image(cv_image)

        # Display the processed image
        cv2.imshow("Processed Undistorted Image", processed_image)
        cv2.waitKey(1)  # Necessary for OpenCV window to update

    def process_image(self, image):
        """
        Perform desired OpenCV operations on the image.

        Arguments:
            image -- Grayscale OpenCV image.

        Returns:
            Processed OpenCV image.
        """
        # Apply Gaussian Blur and Binarization
        blurred_image = cv2.medianBlur(image, 1)
        ret, binary_image = cv2.threshold(blurred_image, 85, 255, cv2.THRESH_BINARY)
        # binary_image = cv2.bitwise_not(binary_image)
        cor_detection = CornerDetection(log_level="DEBUG")
        corner_coords = cor_detection.find(binary_image)

        # Ecken einzeichnen

        # Prepare the Image for Visualization
        processed_image_bgr = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        img_cor = cor_detection.draw(processed_image_bgr)

        # Return the processed image
        return img_cor


def main(args=None):
    """Main function to run the UndistortedImageSubscriber node."""
    rclpy.init(args=args)
    subscriber = CornerDetector()

    try:
        rclpy.spin(subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        subscriber.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
