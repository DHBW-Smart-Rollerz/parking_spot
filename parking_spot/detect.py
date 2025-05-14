from parking_spot.image_operations import DetFilter, getBestMatchingSpots
from parking_spot.square_points import transform_points
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
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


        self.declare_parameter("undistorted_image_topic", "/camera/image/undistorted")
        self.declare_parameter("pose_estimation_topic", "/pose_estimation/pose")


        undistorted_image_topic = (
            self.get_parameter("undistorted_image_topic")
            .get_parameter_value()
            .string_value
        )

        self.subscription = self.create_subscription(
            Image,
            undistorted_image_topic,
            self.listener_callback,
            10,
        )
        self.subscription  # prevent unused variable warning

        self.pose_subscription = self.create_subscription(
            Pose,
            self.get_parameter("pose_estimation_topic"), 
            self.pose_callback,
            10,
        )
        self.pose_subscription  # prevent unused variable warning
        self.latest_pose = None

        self.cor_detection = CornerDetection(log_level="DEBUG")
        # Initialize CvBridge
        self.bridge = CvBridge()

        self.get_logger().info(f"Subscribed to {undistorted_image_topic}")

    def pose_callback(self, msg: Pose):
        self.latest_pose = msg
        self.get_logger().debug(
            f"Received pose: position=({msg.position.x:.2f}, {msg.position.y:.2f}, {msg.position.z:.2f}), "
            f"orientation=({msg.orientation.x:.2f}, {msg.orientation.y:.2f}, {msg.orientation.z:.2f}, {msg.orientation.w:.2f})"
        )


    def listener_callback(self, msg):
        self.state = "detecting"
        """Callback function for the undistorted image subscriber."""
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="8UC1")
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return

    # Process the image using OpenCV
        if (self.state == "detecting"):
            #processed_image = 
            self.detecting(cv_image)

        elif (self.state == "driving_straight"):
            #processed_image = 
            self.driving_straight()

        elif (self.state == "turn_and_drive_spot"):
            #processed_image = 
            self.parking()

        elif (self.state == "unparking"):
            #processed_image = 
            self.unparking()

        elif (self.state == "finished"):
            True
        else:
            #processed_image = 
            self.detecting()

        # Display the processed image
        #cv2.imshow("Processed Undistorted Image", processed_image)
        #cv2.waitKey(1)  # Necessary for OpenCV window to update

    def detecting(self, image):
        """
        Perform desired OpenCV operations on the image.

        Arguments:
            image -- Grayscale OpenCV image.

        Returns:
            Processed OpenCV image.
        """
        # Get coordinates at beginning to calculate later driving after
        # detecting and pos change
        self.detected_coords = [0, 0]

        self.binary_image = DetFilter(image)


        processed_image_bgr = cv2.cvtColor(self.binary_image, cv2.COLOR_GRAY2BGR)

        self.corner_coords = self.cor_detection.find(self.binary_image)
        #img_cor = self.cor_detection.draw(processed_image_bgr, self.corner_coords)

        self.spots, self.spots_w = self.cor_detection.getFullParkingSpots(self.corner_coords)
        
        best_matching_spot, best_matching_spot_w = getBestMatchingSpots(self.spots, self.spots_w, self.binary_image)


        
        #img_cor = self.cor_detection.draw_spots(processed_image_bgr, spots=[best_matching_spot])
        
        # self.chosen_spot = getBestMatchingSpots(self.spots) check for objects in spots/get spot with best matching distances between points
        
        #stop vehicle, recalculate with coordinate since start of iteration
        
        if (best_matching_spot_w is not None):
            self.route, self.route_two_points, img_cor = self.cor_detection.get_draw_route(best_matching_spot_w, img_cor)
            self.route_in_global_global = transform_points(self.latest_pose.position.x, self.latest_pose.position.y, self.latest_pose.orientation.z, self.route_two_points)
            self.state = "driving_straight"
          
        # Return the processed image
        #return img_cor

    def driving_straight(self):
        turning_point = self.route_two_points[0]
        xp_smaller = turning_point[0] < self.latest_pose.position.x
        yp_smaller = turning_point[1] < self.latest_pose.position.y

        while (not (self.latest_pose.position.x - 50 < turning_point[0] < self.latest_pose.position.x + 50) and not ( self.latest_pose.position.y - 50 < turning_point[1] < self.latest_pose.position.y + 50)):    
            
        
        self.state = "turn_and_drive_spot"

    def turn_and_drive_spot(self, image):
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
