import rospy
import rostopic
from geometry_msgs.msg import Point
import numpy as np


class BallTracker:
    def __init__(self):
        # rospy.init_node("ball_tracker", anonymous=True)

        # Camera values
        self.width = 1280
        self.height = 720
        self.x_fov = np.deg2rad(86)
        self.y_fov = np.deg2rad(57)
        self.cTw = np.array(
            [
                [
                    0.3278463,  0.6182624, -0.7143308,
                    1.011,
                ],
                [
                    0.9440518, -0.2430674,  0.2229002,
                    -0.1202,
                ],
                [
                    -0.0358197, -0.7474422, -0.6633604,
                    0.454,
                ],
                [0, 0, 0, 1],
            ]
        )

        # Ball values
        self.ball_radius = 0.02
        self.ball_z = 0.02

        # Subscriber
        TopicType, topic_str, _ = rostopic.get_topic_class("/ball/tracking_update")
        self.subscriber = rospy.Subscriber(
            "/ball/tracking_update", TopicType, self.callback
        )

    def callback(self, data):
        # Compute the line, on which the ball is located
        # angles
        x_ang = (data.x - self.width / 2) / self.width * self.x_fov
        y_ang = (data.y - self.height / 2) / self.height * self.y_fov

        # Homogeneous direction vector from camera perspective
        self.dir = np.array([np.sin(x_ang), np.sin(y_ang), 1, 0]).T
        self.dir = self.dir / np.linalg.norm(self.dir)

        # Rotate the direction vector to the world frame
        self.ball_dir = self.cTw @ self.dir

        self.r = data.r
        # Compute the absolute position
        self.absolute_position = self.compute_absolute_position()

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
        tracker = BallTracker()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
