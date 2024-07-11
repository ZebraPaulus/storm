import rospy
import numpy as np
from sensor_msgs.msg import JointState
from std_msgs.msg import (
    Float64MultiArray,
)  # Assuming joint commands are sent as an array of floats
from copy import deepcopy


class MPCRobotController:
    def __init__(self):
        # Initialize the ROS node (if not already initialized)
        if not rospy.get_node_uri():
            rospy.init_node("mpc_robot_controller", anonymous=True, disable_signals=True)

        # Subscriber for joint states
        self.joint_state_subscriber = rospy.Subscriber(
            "franka_motion_control/joint_states", JointState, self.joint_state_callback
        )
        self.current_joint_state = JointState()

        # Publisher for joint commands
        self.joint_command_publisher = rospy.Publisher(
            "franka_motion_control/joint_command", JointState, queue_size=1
        )

    def joint_state_callback(self, msg):
        """
        Callback function for joint state updates.
        """
        self.current_joint_state = msg

    def get_current_joint_state(self):
        """
        Returns the current joint state.
        """
        cjs = deepcopy(self.current_joint_state)
        current_robot_state = {
                    "name": cjs.name,
                    "position": np.array(cjs.position),
                    "velocity": np.array(cjs.velocity),
                    "acceleration": np.array(cjs.effort),
                }

    def send_joint_command(self, joint_commands):
        """
        Sends a command to the robot's joints.
        :param joint_commands: A list of joint positions/velocities/efforts.
        """
        command_msg = Float64MultiArray()
        command_msg.data = joint_commands
        self.joint_command_publisher.publish(command_msg)
    
    def close(self):
        """
        Close the ROS node.
        """
        self.joint_state_subscriber.unregister()
        self.joint_command_publisher.unregister()
        rospy.signal_shutdown("Shutting down the node.")
        print("Planned shutdown - Goodbye!")

        


# Example usage
if __name__ == "__main__":
    controller = MPCRobotController()
    rospy.spin()  # Keep the node running to listen to callbacks

    # Example on how to access current joint state and send a command
    # current_state = controller.get_current_joint_state()
    # print(current_state)
    # controller.send_joint_command([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])  # Example command
