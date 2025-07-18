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

from hexapod_rqt.hexapod_kinematics import load_hexapod_robot
from hexapod_rqt.hexapod_planner import Polynomial_with_waypoint, LSPB

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

        self.node = node
        self.joint_state_subscriber = self.node.create_subscription(
            JointState, 'joint_states',
            self.joint_state_callback, self.QOS_REKL5V
        )

        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_indicators)
        self.update_timer.start(10)

        self.hexapod = load_hexapod_robot()

        self.lines = {}
        self.scatters = {}
        self.axis = {}

        self.theta_lists = np.zeros((6, 3))
        self.pose = np.zeros((2, 3))
        self.pose[1, -1] = 0.455025253
        self.home_joint_positions = np.array([
            [0.0, -np.deg2rad(45), -np.deg2rad(135)],
            [0.0, -np.deg2rad(45), -np.deg2rad(135)],
            [0.0, -np.deg2rad(45), -np.deg2rad(135)],
            [0.0,  np.deg2rad(45),  np.deg2rad(135)],
            [0.0,  np.deg2rad(45),  np.deg2rad(135)],
            [0.0,  np.deg2rad(45),  np.deg2rad(135)]
        ])
        self.joint_positions = self.home_joint_positions.reshape(18)

        self.trajectory = None
        self.traj_index = 0
        
        self.START = False
        self.STOP = False
        self.HOME = False
        self.UP = False
        self.DOWN = False

        self.initialize_ui()

    def initialize_ui(self):
        self.q_tab_widget : QTabWidget = self.findChild(QTabWidget, 'TabWidget')
        if self.q_tab_widget is None:
            self.node.get_logger().error("No such a name 'TabWidget'.")
            return

        first_tab = QWidget()
        first_tab_layout = QVBoxLayout(first_tab)

        robot_monitor = QGroupBox("Robot Monitor", first_tab)
        robot_monitor.setMinimumSize(600, 600)
        robot_monitor_layout = QVBoxLayout(robot_monitor)

        self.robot_plot_widget = gl.GLViewWidget()
        self.robot_plot_widget.setBackgroundColor(pg.mkColor('k'))
        zgrid = gl.GLGridItem(color=(150, 150, 150, 100))
        self.robot_plot_widget.addItem(zgrid)

        self.hexapod.update_dh_params_list(self.home_joint_positions)
        positions = self.hexapod.forward_kinematics(self.pose)
        for idx, pos in enumerate(positions):
            self.lines[f'leg{idx+1}'] = gl.GLLinePlotItem(pos=pos, color=(0.5, 0.5, 0.5, 1.0), width=5.0)
            self.scatters[f'leg{idx+1}'] = gl.GLScatterPlotItem(pos=pos, size=10.0)
            self.robot_plot_widget.addItem(self.lines[f'leg{idx+1}'])
            self.robot_plot_widget.addItem(self.scatters[f'leg{idx+1}'])

        axis = self.hexapod.robot['leg1'].get_base_axis(self.pose)
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
        self.max_stride_label = QLabel()
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
        self.max_duration_label = QLabel()
        self.max_duration_label.setFixedWidth(80)

        duration_layout.addWidget(self.cur_duration_label)
        duration_layout.addWidget(self.duration_slider)
        duration_layout.addWidget(self.max_duration_label)

        steps_widget = QWidget()
        steps_layout = QHBoxLayout(steps_widget)
        
        self.cur_steps_label = QLabel("Current Steps:    ")
        self.cur_steps_label.setFixedWidth(330)
        self.steps_slider = QSlider(Qt.Horizontal)
        self.steps_slider.valueChanged.connect(
            self.on_steps_slider_changed
        )
        self.max_steps_label = QLabel()
        self.max_steps_label.setFixedWidth(80)

        steps_layout.addWidget(self.cur_steps_label)
        steps_layout.addWidget(self.steps_slider)
        steps_layout.addWidget(self.max_steps_label)

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
        robot_controller_layout.addWidget(steps_widget)
        robot_controller_layout.addWidget(mode_widget)

        first_tab_layout.addWidget(robot_monitor)
        first_tab_layout.addWidget(robot_controller)

        self.q_tab_widget.addTab(first_tab, "Hexapod Robot Controller")

    def on_stride_slider_changed(self, value):
        pass

    def on_duration_slider_changed(self, value):
        pass

    def on_steps_slider_changed(self, value):
        pass

    def on_start_button_clicked(self):
        pass

    def on_stop_button_clicked(self):
        pass

    def on_home_button_clicked(self):
        start_joint_positions_1 = self.joint_positions.copy()
        self.hexapod.update_dh_params_list(start_joint_positions_1.reshape(6, 3))
        start_link_positions_1 = self.hexapod.forward_kinematics(self.pose, is_world=False)[:, -1, :].reshape(6, 3)

        leg_list_1 = [0, 2, 4]
        way_link_positions_1 = start_link_positions_1.copy()
        way_link_positions_1[leg_list_1, -1] += 0.15
        way_joint_positions_1 = self.hexapod.inverse_kinematics(way_link_positions_1)
        
        end_joint_positions_1 = way_joint_positions_1.copy()
        end_joint_positions_1[leg_list_1] = self.home_joint_positions[leg_list_1]
        
        self.hexapod.update_dh_params_list(end_joint_positions_1.reshape(6, 3))
        start_link_positions_2 = self.hexapod.forward_kinematics(self.pose, is_world=False)[:, -1, :].reshape(6, 3)
        start_joint_positions_2 = self.hexapod.inverse_kinematics(start_link_positions_2)
        
        leg_list_2 = [1, 3, 5]
        way_link_positions_2 = start_link_positions_2.copy()
        way_link_positions_2[leg_list_2, -1] += 0.15
        way_joint_positions_2 = self.hexapod.inverse_kinematics(way_link_positions_2)
        
        end_joint_positions_2 = way_joint_positions_2.copy()
        end_joint_positions_2[leg_list_2] = self.home_joint_positions[leg_list_2]

        traj_1 = Polynomial_with_waypoint(
            start_joint_positions_1.reshape(18), way_joint_positions_1.reshape(18), end_joint_positions_1.reshape(18), 2.0
        )
        traj_2 = Polynomial_with_waypoint(
            start_joint_positions_2.reshape(18), way_joint_positions_2.reshape(18), end_joint_positions_2.reshape(18), 2.0
        )
        self.trajectory = np.concatenate((traj_1, traj_2), axis=0)
        self.HOME = True

    def on_up_button_clicked(self):
        pass

    def on_down_button_clicked(self):
        pass

    def joint_state_callback(self, msg):
        self.joint_positions = msg.positions

    def update_indicators(self):
        if self.trajectory is not None:
            self.plot_robot(self.trajectory[self.traj_index].reshape(6, 3))
            self.traj_index += 1

            if self.trajectory.shape[0] <= self.traj_index:
                self.trajectory = None
                self.traj_index = 0
    
    def plot_robot(self, theta_lists):
        self.hexapod.update_dh_params_list(theta_lists)
        positions = self.hexapod.forward_kinematics(self.pose)
        for idx, pos in enumerate(positions):
            self.lines[f'leg{idx+1}'].setData(pos=pos)
            self.scatters[f'leg{idx+1}'].setData(pos=pos)
        
        axis = self.hexapod.robot['leg1'].get_base_axis(self.pose)
        self.axis['x'].setData(pos=axis[0])
        self.axis['y'].setData(pos=axis[1])
        self.axis['z'].setData(pos=axis[2])

    def shutdown_widget(self):
        self.node.destroy_node()