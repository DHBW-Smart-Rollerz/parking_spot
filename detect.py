from parking_spot.image_operations import DetFilter
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
        self.state = "detecting"
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
        while (self.state):
            if (self.state == "detecting"):
                processed_image = self.detecting(cv_image)
            
            elif (self.state == "detected"):
                processed_image = self.detected(cv_image)
                
            elif (self.state == "parking"):
                processed_image = self.parking(cv_image)

            elif (self.state == "unparking"):
                processed_image = self.unparking(cv_image)
                
            elif (self.state == "finished"):
                True
            else:
                processed_image = self.detecting(cv_image)
            
            # Display the processed image
            cv2.imshow("Processed Undistorted Image", processed_image)
            cv2.waitKey(1)  # Necessary for OpenCV window to update

    def detecting(self, image):
        """
        Perform desired OpenCV operations on the image.

        Arguments:
            image -- Grayscale OpenCV image.

        Returns:
            Processed OpenCV image.
        """
        self.binary_image = DetFilter(image)
        cor_detection = CornerDetection(log_level="DEBUG")

        self.detecting_pos = [0, 0] # GET CURRENT POS to do recalc for pos change between detect & stop 
        
        self.corner_coords = cor_detection.find(self.binary_image)

        # [DEBUG] Visualizing
        processed_image_bgr = cv2.cvtColor(self.binary_image, cv2.COLOR_GRAY2BGR)
        img_cor = cor_detection.draw(processed_image_bgr, self.corner_coords)

        # Return the processed image
        return img_cor
    
    def detected(self, image):
        if (True):  # -----condition if first point of parking route is reached
            self.state = "parking"
        return image
    
    def parking(self, image):
        if (True):  # ------condition if parked
            self.state = "unparking"
        return image
    
    def unparking(self, image):
        
        if (True):  # ---condition if unparked, end by giving control back 
            self.state = "finished"
        
        return image


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
