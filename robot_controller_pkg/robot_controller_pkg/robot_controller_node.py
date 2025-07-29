import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from robot_msgs.msg import MotionParameters

import numpy as np
from hexapod_robot_controller.planning import homing, spreading, bouncing, stepping, walking

def generate_trajectory(func, msg):
    motion_params = {'current_pose': np.array([x for x in msg.cur_pose]),
                     'goal_pose': np.array([x for x in msg.goal_pose]),
                     'current_positions': np.array([x for x in msg.cur_positions]),
                     'goal_positions': np.array([x for x in msg.goal_positions]),
                     'value': np.array([x for x in msg.value]),
                     'duration': float(msg.duration)}
    if func.__name__ != 'stepping':
        return func(motion_params)

class RobotControllerNode(Node):
    QOS_REKL5V = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        history=QoSHistory.KEEP_LAST,
        depth=5,
        durability=QoSDurabilityPolicy.VOLATILE
    )
    def __init__(self):
        super().__init__('robot_controller_node')

        self.joint_trajectory_publisher = self.create_publisher(
            JointTrajectory, 'joint_trajectory', self.QOS_REKL5V
        )
        self.point_publisher = self.create_publisher(
            Point, 'point', self.QOS_REKL5V
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

        self.base_traj = None
        self.foot_traj = None
        
        self.total_length = 0
        self.index        = 0

        self.joint_position = None
        self.base_position = None
        
        self.functions = {'homing': homing,
                          'spreading': spreading,
                          'bouncing': bouncing,
                          'stepping': stepping,
                          'walking': walking}

    def joint_state_callback(self, msg):
        self.joint_state_msg = msg

    def motion_parameters_callback(self, msg):
        self.base_traj, self.foot_traj = generate_trajectory(self.functions.get(msg.name), msg)
        self.total_length = self.foot_traj.shape[0]

    def publish_joint_trajecotry(self):
        if self.base_traj is None or self.foot_traj is None:
            return
            
        point     = self.base_traj[self.index]
        joint_pos = self.foot_traj[self.index]
        self.index += 1
       
        base_point = Point()
        base_point.x = point[0]
        base_point.y = point[1]
        base_point.z = point[2]
        self.point_publisher.publish(base_point)

        joint_traj = JointTrajectory()
        joint_traj.joint_names = [name for name in self.joint_state_msg.name]
        joint_traj.points.append(JointTrajectoryPoint())
        joint_traj.points[0].positions = [pos for pos in joint_pos]
        self.joint_trajectory_publisher.publish(joint_traj)
       
        if self.total_length <= self.index:
            self.base_traj = None
            self.foot_traj = None
            self.total_length = 0
            self.index = 0
            
def main(args=None):
    rclpy.init(args=args)
    robot_controller_node = RobotControllerNode()
    rclpy.spin(robot_controller_node)
    robot_controller_node.destroy_node()
    rclpy.shutdown()
    
if __name__ == "__main__":
    main()
