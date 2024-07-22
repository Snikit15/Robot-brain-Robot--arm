import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from control_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from math import pi

class RobotArmControlNode(Node):
    def __init__(self):
        super().__init__('robot_arm_control_node')
        self.cmd_vel_subscription = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )
        self.joint_trajectory_publisher = self.create_publisher(
            JointTrajectory,
            '/joint_arm_position_controller/joint_trajectory',
            10
        )
        self.joint_names = ["base_yaw_joint", "yaw_bicep_joint", "bicep_forearm_joint", "forearm_endeffector_joint"]

    def cmd_vel_callback(self, msg):
        joint_positions = [0.0] * len(self.joint_names)
        joint_velocities = [0.0] * len(self.joint_names)

        # Map linear.x to joint positions and angular.z to joint velocities for each joint
        for i in range(len(self.joint_names)):
            if i == 0:  # Control base_yaw_joint
                joint_positions[i] = msg.linear.x * 0.1  # Adjust the scaling factor as needed
            elif i == 1:  # Control yaw_bicep_joint
                joint_positions[i] = msg.linear.y * 0.1  # Adjust the scaling factor as needed
            elif i == 2:  # Control bicep_forearm_joint
                joint_positions[i] = msg.linear.z * 0.1  # Adjust the scaling factor as needed
            elif i == 3:  # Control forearm_endeffector_joint
                joint_velocities[i] = msg.angular.z * 0.1  # Adjust the scaling factor as needed

        joint_trajectory = JointTrajectory()
        joint_trajectory.joint_names = self.joint_names
        joint_trajectory.points.append(JointTrajectoryPoint(
            positions=joint_positions,
            velocities=joint_velocities,
            time_from_start=rclpy.duration.Duration(seconds=0.1)
        ))

        self.joint_trajectory_publisher.publish(joint_trajectory)

def main(args=None):
    rclpy.init(args=args)
    node = RobotArmControlNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
