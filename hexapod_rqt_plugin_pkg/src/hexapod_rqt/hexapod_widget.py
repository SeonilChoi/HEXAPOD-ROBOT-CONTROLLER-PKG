import os
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from itertools import cycle

from python_qt_binding import loadUi
from python_qt_binding.QtWidgets import QMainWindow, QWidget, QTabWidget, QGroupBox
from python_qt_binding.QtWidgets import QVBoxLayout, QHBoxLayout
from python_qt_binding.QtWidgets import QLabel, QMenu, QAction, QStyle
from python_qt_binding.QtWidgets import QSlider, QToolButton, QPushButton, QPlainTextEdit
from python_qt_binding.QtCore import Qt, QTimer

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from hexapod_rqt.hexapod_kinematics import forward_kinematics_2, get_base_axis
from hexapod_rqt.hexapod_planner import homing, walking

PI = 3.141592653589793

class HexapodWidget(QMainWindow):
    QOS_REKL5V = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=5,
        durability=QoSDurabilityPolicy.VOLATILE
    )
    def __init__(self, node):
        super().__init__()
        ui_file = os.path.join(
            get_package_share_directory('hexapod_rqt_plugin_pkg'),
            'resource',
            'hexapod_rqt.ui'
        )
        try:
            loadUi(ui_file, self)
        except Exception as e:
            rclpy.logging.get_logger('HexapodWidget').error(
                f"{e}"
            )
            raise

        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_indicators)
        self.update_timer.start(20)
        
        self.node = node
        self.joint_state_subscriber = self.node.create_subscription(
            JointState, 'joint_states',
            self.joint_state_callback, self.QOS_REKL5V
        )
        self.joint_trajectory_publisher = self.node.create_publisher(
            JointTrajectory, 'joint_trajectory', self.QOS_REKL5V
        )

        self.lines = {}
        self.scatters = {}
        self.axis = {}

        self.pose = np.zeros((2, 3))
        self.pose[1, -1] = 0.455025253
        self.home_joint_positions = np.array([
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            -np.deg2rad(45), -np.deg2rad(45), -np.deg2rad(45),
             np.deg2rad(45),  np.deg2rad(45),  np.deg2rad(45),
            -np.deg2rad(135), -np.deg2rad(135), -np.deg2rad(135),
             np.deg2rad(135),  np.deg2rad(135),  np.deg2rad(135)
        ])
        
        self.joint_positions = None
        self.joint_name = None

        self.trajectory = None
        self.base_trajectory = None
        self.traj_index = 0
        
        self.stride   = 0
        self.duration = 0
        self.goal_x   = 0
        self.goal_y   = 0
        self.goal_z   = 0

        self.MOVING = False
        self.initialize_ui()

    def initialize_ui(self):
        self.q_tab_widget : QTabWidget = self.findChild(QTabWidget, 'TabWidget')
        if self.q_tab_widget is None:
            self.node.get_logger().error("No such a name 'TabWidget'.")
            return

        first_tab = QWidget()
        first_tab_layout = QVBoxLayout(first_tab)

        robot_monitor = QGroupBox("Robot Monitor", first_tab)
        robot_monitor_layout = QVBoxLayout(robot_monitor)

        self.robot_plot_widget = gl.GLViewWidget()
        self.robot_plot_widget.setMinimumSize(300, 500)
        self.robot_plot_widget.setBackgroundColor(pg.mkColor('k'))
        zgrid = gl.GLGridItem(color=(150, 150, 150, 100))
        self.robot_plot_widget.addItem(zgrid)

        positions = forward_kinematics_2(self.pose, self.home_joint_positions, True)
        for idx, pos in enumerate(positions):
            self.lines[f'leg{idx+1}'] = gl.GLLinePlotItem(
                pos=pos, color=(0.5, 0.5, 0.5, 1.0), width=5.0)
            self.scatters[f'leg{idx+1}'] = gl.GLScatterPlotItem(pos=pos, size=10.0)
            self.robot_plot_widget.addItem(self.lines[f'leg{idx+1}'])
            self.robot_plot_widget.addItem(self.scatters[f'leg{idx+1}'])

        axis = get_base_axis(self.pose)
        self.axis['x'] = gl.GLLinePlotItem(pos=axis[0], color=(1.0, 0.0, 0.0, 1.0), width=5.0)
        self.axis['y'] = gl.GLLinePlotItem(pos=axis[1], color=(0.0, 1.0, 0.0, 1.0), width=5.0)
        self.axis['z'] = gl.GLLinePlotItem(pos=axis[2], color=(0.0, 0.0, 1.0, 1.0), width=5.0)
        self.robot_plot_widget.addItem(self.axis['x'])
        self.robot_plot_widget.addItem(self.axis['y'])
        self.robot_plot_widget.addItem(self.axis['z'])

        robot_monitor_layout.addWidget(self.robot_plot_widget)

        robot_controller = QGroupBox("Robot Controller", first_tab)
        robot_controller_layout = QVBoxLayout(robot_controller)

        stride_widget = QWidget()
        stride_layout = QHBoxLayout(stride_widget)

        self.cur_stride_label = QLabel("Current Stride:    ")
        self.cur_stride_label.setFixedWidth(330)
        self.stride_slider = QSlider(Qt.Horizontal)
        self.stride_slider.valueChanged.connect(
            self.on_stride_slider_changed
        )
        self.stride_slider.setRange(0, 20)
        self.max_stride_label = QLabel("0.2 m")
        self.max_stride_label.setFixedWidth(80)

        stride_layout.addWidget(self.cur_stride_label)
        stride_layout.addWidget(self.stride_slider)
        stride_layout.addWidget(self.max_stride_label)

        duration_widget = QWidget()
        duration_layout = QHBoxLayout(duration_widget)
        
        self.cur_duration_label = QLabel("Current Duration:    ")
        self.cur_duration_label.setFixedWidth(330)
        self.duration_slider = QSlider(Qt.Horizontal)
        self.duration_slider.valueChanged.connect(
            self.on_duration_slider_changed
        )
        self.duration_slider.setRange(0, 50)
        self.max_duration_label = QLabel("10 s")
        self.max_duration_label.setFixedWidth(80)

        duration_layout.addWidget(self.cur_duration_label)
        duration_layout.addWidget(self.duration_slider)
        duration_layout.addWidget(self.max_duration_label)

        position_widget = QWidget()
        position_layout = QHBoxLayout(position_widget)
        
        self.cur_position_label = QLabel("Current Position(m): ")
        self.cur_position_label.setFixedWidth(300)
        self.cur_x_label = QLabel("0.0")
        self.cur_x_label.setMaximumSize(80, 20)
        self.cur_y_label = QLabel("0.0")
        self.cur_y_label.setMaximumSize(80, 20)
        self.cur_z_label = QLabel("0.0")
        self.cur_z_label.setMaximumSize(80, 20)
        self.goal_position_label = QLabel("Goal Position(m): ")
        self.goal_position_label.setFixedWidth(300)
        self.goal_x_text = QPlainTextEdit()
        self.goal_x_text.setMaximumSize(80, 30)
        self.goal_y_text = QPlainTextEdit()
        self.goal_y_text.setMaximumSize(80, 30)
        self.goal_z_text = QPlainTextEdit()
        self.goal_z_text.setMaximumSize(80, 30)
        
        position_layout.addWidget(self.cur_position_label)
        position_layout.addWidget(self.cur_x_label)
        position_layout.addWidget(self.cur_y_label)
        position_layout.addWidget(self.cur_z_label)
        position_layout.addWidget(self.goal_position_label)
        position_layout.addWidget(self.goal_x_text)
        position_layout.addWidget(self.goal_y_text)
        position_layout.addWidget(self.goal_z_text)

        mode_widget = QWidget()
        mode_layout = QHBoxLayout(mode_widget)

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(
            self.on_start_button_clicked
        )
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(
            self.on_stop_button_clicked
        )
        self.stop_button.setDisabled(True)
        self.home_button = QPushButton("Home")
        self.home_button.clicked.connect(
            self.on_home_button_clicked
        )
        self.up_button = QPushButton("Up")
        self.up_button.clicked.connect(
            self.on_up_button_clicked
        )
        self.down_button = QPushButton("Down")
        self.down_button.clicked.connect(
            self.on_down_button_clicked
        )

        mode_layout.addWidget(self.start_button)
        mode_layout.addWidget(self.stop_button)
        mode_layout.addWidget(self.home_button)
        mode_layout.addWidget(self.up_button)
        mode_layout.addWidget(self.down_button)

        robot_controller_layout.addWidget(stride_widget)
        robot_controller_layout.addWidget(duration_widget)
        robot_controller_layout.addWidget(position_widget)
        robot_controller_layout.addWidget(mode_widget)

        first_tab_layout.addWidget(robot_monitor)
        first_tab_layout.addWidget(robot_controller)

        self.q_tab_widget.addTab(first_tab, "Hexapod Robot Controller")

    def on_stride_slider_changed(self, value):
        self.stride = value * 0.01
        self.cur_stride_label.setText(f"Current Stride:   {self.stride:.2f} m")

    def on_duration_slider_changed(self, value):
        self.duration = value * 0.2
        self.cur_duration_label.setText(f"Current Duration:   {self.duration:.1f} s")

    def on_start_button_clicked(self):
        if self.stride == 0 or self.duration == 0:
            return

        if self.goal_x == "" or self.goal_y == "" or self.goal_z == "":
            return
        
        self.moving(True)
        
        cur_pose = self.pose[1].copy()
        cur_pose[-1] = 0.0
        goal_pose = np.array([float(self.goal_x), float(self.goal_y), float(self.goal_z)])
        
        self.trajectory, self.base_trajectory, real_goal_pose = walking(
             self.home_joint_positions, cur_pose, goal_pose, self.stride, self.duration)
        self.base_trajectory[:, -1] *= -1

    def on_stop_button_clicked(self):
        self.trajectory = None
        self.base_trajectory = None
        self.traj_index = 0

        self.moving(False)
        
    def on_home_button_clicked(self):
        self.moving(True)

        cur_pose = self.pose[1].copy()
        cur_pose[-1] = 0.0
        self.trajectory, self.base_trajectory = homing(
            self.home_joint_positions, self.home_joint_positions, cur_pose, 4.0)
        self.base_trajectory[:, -1] *= -1
        self.HOME = True

    def on_up_button_clicked(self):
        pass

    def on_down_button_clicked(self):
        pass

    def joint_state_callback(self, msg):
        self.joint_names = np.array([name for name in msg.name])
        self.joint_positions = np.array([pos for pos in msg.position])
        self.joint_positions = self.joint_positions.reshape(6, 3)
        self.joint_positions = self.joint_positions.T
        self.joint_positions = self.joint_positions.reshape(18)

    def update_indicators(self):
        self.goal_x = self.goal_x_text.toPlainText()
        self.goal_y = self.goal_y_text.toPlainText()
        self.goal_z = self.goal_z_text.toPlainText()

        if self.trajectory is not None:
            joint_positions = self.trajectory[self.traj_index]
            self.pose[1] = self.base_trajectory[self.traj_index].copy()
            self.plot_robot(joint_positions)
            self.traj_index += 1
            
            """
            traj = JointTrajectory()
            traj.joint_names = [name for name in self.joint_names]
            point = JointTrajectoryPoint()
            msg_joint_positions = joint_positions.reshape(3, 6)
            msg_joint_positions = msg_joint_positions.T
            msg_joint_positions = msg_joint_positions.reshape(18)
            point.positions = [pos for pos in msg_joint_positions]
            traj.points.append(point)
            self.joint_trajectory_publisher.publish(traj)
            """
            
            self.cur_x_label.setText(f"{self.pose[1, 0]:.3f}")
            self.cur_y_label.setText(f"{self.pose[1, 1]:.3f}")
            self.cur_z_label.setText(f"{self.pose[1, 2]:.3f}")

            if self.trajectory.shape[0] <= self.traj_index:
                self.MOVING = False

                self.trajectory = None
                self.base_trajectory = None
                self.traj_index = 0
                
                self.moving(False)

    def plot_robot(self, theta_lists):
        positions = forward_kinematics_2(self.pose, theta_lists, True)
        for idx, pos in enumerate(positions):
            self.lines[f'leg{idx+1}'].setData(pos=pos)
            self.scatters[f'leg{idx+1}'].setData(pos=pos)
        
        axis = get_base_axis(self.pose)
        self.axis['x'].setData(pos=axis[0])
        self.axis['y'].setData(pos=axis[1])
        self.axis['z'].setData(pos=axis[2])

    def moving(self, flag):
        if flag:
            self.stride_slider.setDisabled(True)
            self.duration_slider.setDisabled(True)
            self.start_button.setDisabled(True)
            self.stop_button.setEnabled(True)
            self.home_button.setDisabled(True)
            self.up_button.setDisabled(True)
            self.down_button.setDisabled(True)
            self.goal_x_text.setDisabled(True)
            self.goal_y_text.setDisabled(True)
            self.goal_z_text.setDisabled(True)
        else:
            self.stride_slider.setEnabled(True)
            self.duration_slider.setEnabled(True)
            self.start_button.setEnabled(True)
            self.stop_button.setDisabled(True)
            self.home_button.setEnabled(True)
            self.up_button.setEnabled(True)
            self.down_button.setEnabled(True)
            self.goal_x_text.setEnabled(True)
            self.goal_y_text.setEnabled(True)
            self.goal_z_text.setEnabled(True)
            self.goal_x_text.setPlainText("")
            self.goal_y_text.setPlainText("")
            self.goal_z_text.setPlainText("")

    def shutdown_widget(self):
        self.node.destroy_node()
