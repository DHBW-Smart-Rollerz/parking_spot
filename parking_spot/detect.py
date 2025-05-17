from parking_spot.image_operations import DetFilter, getBestMatchingSpots
from parking_spot.square_points import has_turned_90_degrees, transform_points
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Int16, Float32
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from itertools import combinations

from parking_spot.corner_detection import CornerDetection
from camera_preprocessing.transformation.coordinate_transform import (
    CoordinateTransform,
)

speed_straight = 0.40  # speed driving straight until turning point
steering_angle_turn = -35  # steering angle to turn from street to spot
speed_to_spot = 0.03  # speed while turning and driving into spot
tolerance_slow_down = 50  # tolerance in turning point area where vehicle slows down
tolerance_turn = 10  # tolerance at turning point and verhicle turns
tolerance_parking = 20  # tolerance to center of parking spot
factor_backwards = -1  # factor to drive backwards


class CornerDetector(Node):
    """A ROS2 node that subscribes to the undistorted image
    topic and processes the images."""

    def __init__(self):
        self.state = "detecting"
        super().__init__("undistorted_image_subscriber")

        self.speed_publisher = self.create_publisher(
            Float32, "/control/speed/target", 10
        )
        self.steering_publisher = self.create_publisher(
            Int16, "/control/steering_angle/target", 10
        )
        self.control_publisher = self.create_publisher(
            Bool, "/parkingspot/use_control", 10
        )

        self.declare_parameter("undistorted_image_topic", "/camera/image/undistorted")
        self.declare_parameter("pose_estimation_topic", "/pose_estimation/pose")

        undistorted_image_topic = (
            self.get_parameter("undistorted_image_topic")
            .get_parameter_value()
            .string_value
        )

        pose_estimation_topic = (
            self.get_parameter("pose_estimation_topic")
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
            pose_estimation_topic,
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
        """Callback function for the undistorted image subscriber."""
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="8UC1")
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return

        # Process the image using OpenCV
        if self.state == "detecting":
            #
            self.get_logger().info("state: detecting")
            processed_image = self.detecting(cv_image)

        elif self.state == "driving_straight":
            # processed_image =
            self.get_logger().info("state: driving_straight")
            processed_image = self.driving_straight()

        elif self.state == "turn_and_drive_spot":
            # processed_image =
            self.get_logger().info("state: turn_and_drive_spot")
            processed_image = self.turn_and_drive_spot()

        elif self.state == "unparking":
            # processed_image =
            self.get_logger().info("state: unparking")
            processed_image = self.unparking()

        elif self.state == "finished":
            self.get_logger().info("state: finished")
            True
        else:
            # processed_image =
            self.get_logger().info("else state")
            self.detecting()

        # Display the processed image
        try:
            cv2.imshow("Processed Undistorted Image", processed_image)
            cv2.waitKey(1)  # Necessary for OpenCV window to update
        except Exception as e:
            e

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
        if self.latest_pose is None:
            self.latest_pose = Pose()
            self.latest_pose.position.x = 0
            self.latest_pose.position.y = 0
            self.latest_pose.orientation.z = 0
            self.get_logger().info("No pose received yet, using default values.")
        self.binary_image = DetFilter(image)

        processed_image_bgr = cv2.cvtColor(self.binary_image, cv2.COLOR_GRAY2BGR)

        self.corner_coords = self.cor_detection.find(self.binary_image)
        # img_cor = self.cor_detection.draw(processed_image_bgr, self.corner_coords)
        self.spots, self.spots_w = self.cor_detection.getFullParkingSpots(
            self.corner_coords
        )

        best_matching_spot, best_matching_spot_w = getBestMatchingSpots(
            self.spots, self.spots_w, self.binary_image
        )

        img_cor = self.cor_detection.draw_spots(
            processed_image_bgr, spots=[best_matching_spot]
        )

        # stop vehicle, recalculate with coordinate since start of iteration

        if best_matching_spot_w is not None:
            self.route, self.route_three_points, img_cor = (
                self.cor_detection.get_draw_route(best_matching_spot_w, img_cor)
            )
            self.route_in_global_global = transform_points(
                self.latest_pose.position.x,
                self.latest_pose.position.y,
                self.latest_pose.orientation.z,
                self.route_three_points,
            )
            print("switch to driving straight")
            self.state = "driving_straight"
            self.img_cor = img_cor

        # Return the processed image
        return img_cor

    def driving_straight(self):
        turning_point = self.route_three_points[0]
        print(turning_point, "turning point")
        print(self.latest_pose.position.x, self.latest_pose.position.y)
        self.publish_bool(self.control_publisher, True)  # tale control over vehicle
        speed = speed_straight
        print(
            self.latest_pose.position.x - tolerance_slow_down
            < turning_point[0]
            < self.latest_pose.position.x + tolerance_slow_down
        )
        print(
            self.latest_pose.position.y - tolerance_slow_down
            < turning_point[1]
            < self.latest_pose.position.y + tolerance_slow_down
        )
        if not (
            (
                self.latest_pose.position.x - tolerance_slow_down
                < turning_point[0]
                < self.latest_pose.position.x + tolerance_slow_down
            )
            and (
                self.latest_pose.position.y - tolerance_slow_down
                < turning_point[1]
                < self.latest_pose.position.y + tolerance_slow_down
            )
        ):
            speed = speed_straight  # set speed to drive straight
        elif not (
            (
                self.latest_pose.position.x - tolerance_turn
                < turning_point[0]
                < self.latest_pose.position.x + tolerance_turn
            )
            and (
                self.latest_pose.position.y - tolerance_turn
                < turning_point[1]
                < self.latest_pose.position.y + tolerance_turn
            )
        ):
            speed = speed_to_spot  # set speed to park
        else:
            speed = speed_to_spot  # set to zero if reaction to slow
            self.rad_before_turn = self.latest_pose.orientation.z
            self.state = "turn_and_drive_spot"
        self.publish_float(self.speed_publisher, speed)
        print(speed, "speed")
        return self.img_cor

    def turn_and_drive_spot(self):
        print(self.route_three_points[2])
        print(self.latest_pose.position.x, self.latest_pose.position.y)
        if not has_turned_90_degrees(
            self.rad_before_turn,
            self.latest_pose.orientation.z,
            tolerance_deg=tolerance_turn,
        ):
            steering = steering_angle_turn
            speed = speed_to_spot

        elif not (
            (
                self.latest_pose.position.x - tolerance_parking
                < self.route_three_points[2][0]
                < self.latest_pose.position.x + tolerance_parking
            )
            and (
                self.latest_pose.position.y - tolerance_parking
                < self.route_three_points[2][1]
                < self.latest_pose.position.y + tolerance_parking
            )
        ):
            steering = 0
            speed = speed_to_spot
        else:
            self.rad_before_turn = self.latest_pose.orientation.z
            steering = 0
            speed = 0
            self.state = "unparking"

        self.publish_int(self.steering_publisher, steering)
        self.publish_float(self.speed_publisher, speed)
        print(steering, "steering", speed, "speed")
        return self.img_cor

    def unparking(self):
        print(self.route_three_points[1])
        print(self.latest_pose.position.x, self.latest_pose.position.y)
        if not (
            (
                self.latest_pose.position.x - tolerance_parking
                < self.route_three_points[1][0]
                < self.latest_pose.position.x + tolerance_parking
            )
            and (
                self.latest_pose.position.y - tolerance_parking
                < self.route_three_points[1][1]
                < self.latest_pose.position.y + tolerance_parking
            )
        ):
            steering = 0
            speed = factor_backwards * speed_to_spot
        elif not has_turned_90_degrees(
            self.rad_before_turn,
            self.latest_pose.orientation.z,
            tolerance_deg=tolerance_turn,
        ):
            steering = steering_angle_turn
            speed = factor_backwards * speed_to_spot
        else:
            steering = 0
            speed = 0
            self.publish_bool(self.control_publisher, False)
            self.state = "finished"

        self.publish_int(self.steering_publisher, steering)
        self.publish_float(self.speed_publisher, speed)
        print(steering, "steering", speed, "speed")

        return self.img_cor

    def publish_bool(self, publisher, value: bool):
        msg = Bool()
        msg.data = value
        publisher.publish(msg)

    def publish_int(self, publisher, value: int):
        msg = Int16()
        msg.data = value
        publisher.publish(msg)

    def publish_float(self, publisher, value: float):
        msg = Float32()
        msg.data = value
        publisher.publish(msg)


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
