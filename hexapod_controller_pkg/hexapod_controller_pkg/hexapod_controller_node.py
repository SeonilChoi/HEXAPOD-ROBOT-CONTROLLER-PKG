import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from hexapod_robot_msgs.msg import MotionParameters

class HexapodControllerNode(Node):
    QOS_REKL5V = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        history=QoSHistory.KEEP_LAST,
        depth=5,
        durability=QoSDurabilityPolicy.VOLATILE
    )
    def __init__(self):
        super().__init__('hexapod_controller_node')

        self.joint_trajectory_publisher = self.create_publisher(
            JointTrajectory, 'joint_trajectory', self.QOS_REKL5V
        )
        self.joint_state_subscriber = self.create_subscription(
            JointState, 'joint_state',
            self.joint_state_callback, self.QOS_REKL5V
        )
        self.motion_parameters_subscriber = self.create_subscription(
            MotionParameters, 'motion_parameters',
            self.motion_parameters_callback, self.QOS_REKL5V
        )
        self.timer = self.create_timer(
            0.02, self.publish_joint_trajectory
        )

        self.joint_state_msg = JointState()
        self.motion_parameters_msg = MotionParameters()

    def joint_state_callback(self, msg):
        self.joint_state_msg = msg

    def motion_parameters_callback(self, msg):
        self.motion_parameters = msg

    def publish_joint_trajecotry(self):
        pass
