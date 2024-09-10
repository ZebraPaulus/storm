import rospy
import numpy as np
import sys
sys.path.append("/home/dgrawe/ws/wrapper_workspace/devel/lib/python3/dist-packages")
from franka_gripper.msg import MoveActionGoal, MoveGoal, GraspActionGoal, GraspGoal, GraspEpsilon
from sensor_msgs.msg import JointState
from actionlib_msgs.msg import GoalID
from copy import deepcopy

class MPCRobotController:
    def __init__(self):
        # Initialize the ROS node (if not already initialized)
        # Subscriber for joint states
        self.available = False
        self.joint_state_subscriber = rospy.Subscriber(
            "franka_motion_control/joint_states", JointState, self.joint_state_callback
        )
        self.current_joint_state = JointState()

        # Publisher for joint commands
        self.joint_command_publisher = rospy.Publisher(
            "franka_motion_control/joint_command", JointState, queue_size=1
        )

        self.gripper_mover_publisher = rospy.Publisher(
            "/franka_gripper/move/goal", MoveActionGoal, queue_size=1
        )

        self.gripper_grasp_publisher = rospy.Publisher(
            "/franka_gripper/grasp/goal", GraspActionGoal, queue_size=1
        )
        self.id = GoalID()
        self.gripped = False

    def joint_state_callback(self, msg):
        """
        Callback function for joint state updates.
        """
        self.available = True
        self.current_joint_state = msg

    def get_current_joint_state(self):
        """
        Returns the current joint state.
        """
        while not self.available:
            print("Waiting for joint state...")
            rospy.sleep(0.1)

        cjs = deepcopy(self.current_joint_state)
        current_robot_state = {
                    "name": cjs.name,
                    "position": np.array(cjs.position),
                    "velocity": np.array(cjs.velocity),
                    "acceleration": np.array(cjs.effort),
                }
        return current_robot_state

    def send_joint_command(self, q_des, qd_des=np.zeros(7), qdd_des=np.zeros(7)):
        """
        Sends a command to the robot's joints.
        :param joint_commands: A list of joint positions/velocities/efforts.
        """
        js = JointState(name=self.current_joint_state.name, position=q_des, velocity=qd_des, effort=qdd_des)
        self.joint_command_publisher.publish(js)

    def grip(self):
        """
        Sends a command to the robot's gripper.
        :param width: The width of the gripper.
        :param speed: The speed of the gripper.
        :param force: The force of the gripper.
        """
        if self.gripped:
            return
        epsilon = GraspEpsilon(inner = 0.03, outer = 0.03)
        goal = GraspGoal(width = 0.03, speed = 1.0, force = 1.0, epsilon = epsilon)

        self.gripper_grasp_publisher.publish(GraspActionGoal(goal=goal))
        self.gripped = True


    def release(self):
        if self.gripped:
            goal = MoveGoal(width = 0.07, speed = 1.0)
            self.gripper_mover_publisher.publish(MoveActionGoal(goal=goal))
            self.gripped = False
    
    def close(self):
        self.gripper_grasp_publisher.unregister()
        self.gripper_mover_publisher.unregister()
        self.joint_command_publisher.unregister()

    def __del__(self):
        """
        Close the ROS node.
        """
        self.joint_state_subscriber.unregister()
        self.joint_command_publisher.unregister()

# Example usage
if __name__ == "__main__":
    rospy.init_node("mpc_robot_controller")
    controller = MPCRobotController()
    # rospy.spin()  # Keep the node running to listen to callbacks
    while True:
        i = input("g for grip, r for release")
        if i == "g":
            controller.grip()
        elif i == "r":
            controller.release()
        else:
            rospy.signal_shutdown("Shutting down...")
            break
    # Example on how to access current joint state and send a command
    # current_state = controller.get_current_joint_state()
    # print(current_state)
    # controller.send_joint_command([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])  # Example command
