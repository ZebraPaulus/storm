import rospy
import rostopic
import numpy as np
import cv2
import copy
from collections import deque
from scipy.interpolate import interp1d


class BallTracker:
    def __init__(self):
        # rospy.init_node("ball_tracker", anonymous=True)
        top = rostopic.get_topic_class("/camera/color/camera_info")
        camera_info = rospy.wait_for_message("/camera/color/camera_info", top[0])
        # Camera values#
        self.width = camera_info.width
        self.height = camera_info.height
        self.D = copy.deepcopy(camera_info.D)
        self.K = copy.deepcopy(np.array(camera_info.K).reshape(3, 3))
        self.R = copy.deepcopy(np.array(camera_info.R).reshape(3, 3))
        self.P = copy.deepcopy(np.array(camera_info.P).reshape(3, 4))

        self.cTw = np.array(
            [
                [
                    0.3463431134015140,
                    0.6351211170161396,
                    -0.6904111923480942,
                    1.052321073423853,
                ],
                [
                    0.9378501564944148,
                    -0.2516690257335052,
                    0.2389556139741229,
                    -0.1766565658459666,
                ],
                [
                    -0.02198935566923014,
                    -0.7302628760977237,
                    -0.682812272905762,
                    0.5364572141822066,
                ],
                [0, 0, 0, 1],
            ]
        )

        # Ball values
        self.ball_radius = 0.02
        self.ball_z = 0.02

        self.history = deque(maxlen=100)

        # Subscriber
        TopicType, topic_str, _ = rostopic.get_topic_class("/ball/tracking_update")
        self.subscriber = rospy.Subscriber(
            "/ball/tracking_update", TopicType, self.callback
        )

    def __del__(self):
        self.subscriber.unregister()

    def callback(self, data):
        # Compute the line, on which the ball is located
        # undistort coordinates
        if data.r < 10:
            return

        raw_points = np.array([[data.x, data.y]], dtype=np.float32)

        # Undistort the points
        undistorted_points = cv2.undistortPoints(
            raw_points, self.K, self.D, R=self.R, P=self.P
        )

        # Convert back to pixel coordinates
        self.p = undistorted_points[0][0]

        # calculate direction vector
        self.ball_dir = self.cTw @ np.array(
            [
                (self.p[0] - self.K[0][2]) / self.K[0][0],
                (self.p[1] - self.K[1][2]) / self.K[1][1],
                1,
                0,
            ]
        )

        # Compute the absolute position
        self.absolute_position = self.compute_absolute_position()
        self.history.append((data.header.stamp.to_sec(), self.absolute_position))
        self.compute_time_pos()

    def compute_time_pos(self):
        # Extract timestamps and positions from history
        times = [entry[0] for entry in self.history]
        positions = [entry[1] for entry in self.history]

        # Create interpolation functions for each coordinate
        self.interp_x = interp1d(
            times, [pos[0] for pos in positions], fill_value="extrapolate"
        )
        self.interp_y = interp1d(
            times, [pos[1] for pos in positions], fill_value="extrapolate"
        )
        self.interp_z = interp1d(
            times, [pos[2] for pos in positions], fill_value="extrapolate"
        )

    def get_position_at_time(self, timestamp):
        # Use the interpolation functions to get the position at the given timestamp
        x = self.interp_x(timestamp)
        y = self.interp_y(timestamp)
        z = self.interp_z(timestamp)
        return np.array([x, y, z])

    def compute_absolute_position(self):
        # Line equation: point + t * direction_vector
        point = self.cTw @ np.array([0, 0, 0, 1])
        direction = self.ball_dir

        # Find t where the line intersects the plane z = self.plane_z
        t = (self.ball_z - point[2]) / direction[2]

        # Compute the intersection point
        intersection_point = point + t * direction

        # Adjust for ball radius
        absolute_position = np.zeros(3)
        absolute_position[0] = intersection_point[0]
        absolute_position[1] = intersection_point[1]
        absolute_position[2] = intersection_point[2]

        return absolute_position

    def get_absolute_position(self):
        while not hasattr(self, "absolute_position"):
            print("Waiting for ball position...")
            rospy.sleep(0.1)

        return self.absolute_position


if __name__ == "__main__":
    try:
        rospy.init_node("ball_tracker", anonymous=True)
        tracker = BallTracker()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
