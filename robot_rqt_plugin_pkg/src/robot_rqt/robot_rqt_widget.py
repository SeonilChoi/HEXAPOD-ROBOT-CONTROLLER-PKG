import os
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from python_qt_binding import loadUi
from python_qt_binding.QtWidgets import QMainWindow, QWidget, QTabWidget, QGroupBox
from python_qt_binding.QtWidgets import QVBoxLayout, QHBoxLayout
from python_qt_binding.QtWidgets import QMenu, QAction, QStyle
from python_qt_binding.QtWidgets import QLabel
from python_qt_binding.QtWidgets import QSlider, QToolButton, QRadioButton, QPushButton, QPlainTextEdit
from python_qt_binding.QtCore import Qt, QTimer

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from robot_msgs.msg import RobotState

import numpy as np
from hexapod_robot_kinematics.kinematics import forward_kinematics, inverse_kinematics, get_base_axis
from hexapod_robot_kinematics.planning import homing, spreading, bouncing, stepping, walking

class RobotRQtWidget(QMainWindow):
    QOS_REKL5V = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=5,
        durability=QoSDurabilityPolicy.VOLATILE
    )
    def __init__(self, node):
        super().__init__()
        
        self.node = node

        ui_file = os.path.join(
            get_package_share_directory('robot_rqt_plugin_pkg'),
            'resource',
            'robot_rqt.ui'
        )
        try:
            loadUi(ui_file, self)
        except Exception as e:
            self.node.get_logger().error(f'[RobotRQtWidget::init] Failed to load ui file: {e}')
            raise

        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_indicators)
        self.update_timer.start(20)

        self.joint_trajectory_publisher = self.node.create_publisher(
            JointTrajectory, 'joint_trajectory', self.QOS_REKL5V
        )
        self.robot_state_subscriber = self.node.create_subscription(
            RobotState, 'robot_state',
            self.robot_state_callback, self.QOS_REKL5V
        )

        self.lines = {}
        self.scatters = {}
        self.axis = {}

        self.home_pose = np.array([[0, 0, 0], [0, 0, 5.45129330e-01]])
        self.home_joint_positions = np.array([0.0, -np.deg2rad(45), np.deg2rad(135),
                                              0.0, -np.deg2rad(45), np.deg2rad(135),
                                              0.0, -np.deg2rad(45), np.deg2rad(135),
                                              0.0,  np.deg2rad(45), -np.deg2rad(135),
                                              0.0,  np.deg2rad(45), -np.deg2rad(135),
                                              0.0,  np.deg2rad(45), -np.deg2rad(135)])

        self.base_traj = None
        self.foot_traj = None
        self.traj_idx  = 0
        
        self.motion_params = {'current_pose': self.home_pose,
                              'goal_pose': np.zeros((2, 3)),
                              'current_positions': self.home_joint_positions,
                              'goal_positions': np.zeros((18,)),
                              'value': np.zeros((3,)),
                              'duration': 0.0}

        self.is_plotting = True
        self.is_moving   = True
        self.mode        = None
        self.value       = np.zeros(3)
        self.duration    = 0.0
        self.goal_x      = 0.0
        self.goal_y      = 0.0

        self.functions = {'homing': homing,
                          'spreading': spreading,
                          'bouncing': bouncing,
                          'walking': walking}

        self.initialize_ui()

    def initialize_ui(self):

        # First Tab
        first_tab = QWidget()

        # Robot Monitor
        robot_monitor = QGroupBox('Robot Monitor', first_tab)
        
        # Plotting Button
        plotting_btn = QRadioButton('Plotting')
        plotting_btn.setChecked(True)
        plotting_btn.clicked.connect(
            self.on_plotting_button_clicked
        )

        # Plot Widget
        plot_widget = gl.GLViewWidget()
        plot_widget.setMinimumSize(300, 500)
        plot_widget.setBackgroundColor(pg.mkColor('k'))
        plot_widget.addItem(gl.GLGridItem(color=(150, 150, 150, 100)))
        self.initialize_plot_widget(plot_widget)

        # Robot Monitor Layout
        robot_monitor_layout = QVBoxLayout(robot_monitor)
        robot_monitor_layout.addWidget(plotting_btn, 0, Qt.AlignRight)
        robot_monitor_layout.addWidget(plot_widget)

        # Robot Controller
        robot_controller = QGroupBox('Robot Controller', first_tab)

        # Mode Widget
        mode_widget = QWidget()

        # Homing Button
        homing_btn = QRadioButton('Homing')
        homing_btn.clicked.connect(
            self.on_homing_button_clicked
        )

        # Spreading Button
        spreading_btn = QRadioButton('Spreading')
        spreading_btn.clicked.connect(
            self.on_spreading_button_clicked
        )
        
        # Bouncing Button
        bouncing_btn = QRadioButton('Bouncing')
        bouncing_btn.clicked.connect(
            self.on_bouncing_button_clicked
        )
        
        # Walking Button
        walking_btn = QRadioButton('Walking')
        walking_btn.clicked.connect(
            self.on_walking_button_clicked
        )

        # Mode Layout
        mode_widget_layout = QHBoxLayout(mode_widget)
        mode_widget_layout.addWidget(homing_btn)
        mode_widget_layout.addWidget(spreading_btn)
        mode_widget_layout.addWidget(bouncing_btn)
        mode_widget_layout.addWidget(walking_btn)
        
        # Value Widget
        value_widget = QWidget()

        # Current Value Label
        self.cur_value_label = QLabel('Current Value:    ')
        self.cur_value_label.setFixedWidth(330)

        # Value Slider
        self.value_slider = QSlider(Qt.Horizontal)
        self.value_slider.setRange(0, 40)
        self.value_slider.valueChanged.connect(
            self.on_value_slider_changed
        )

        # Max Value Label
        max_value_label = QLabel('0.4 m')
        max_value_label.setFixedWidth(80)

        # Value Layout
        value_layout = QHBoxLayout(value_widget)
        value_layout.addWidget(self.cur_value_label)
        value_layout.addWidget(self.value_slider)
        value_layout.addWidget(max_value_label)

        # Duration Widget
        duration_widget = QWidget()

        # Current Duration Label
        self.cur_duration_label = QLabel('Current Duration:    ')
        self.cur_duration_label.setFixedWidth(330)

        # Duration Slider
        self.duration_slider = QSlider(Qt.Horizontal)
        self.duration_slider.setRange(0, 50)
        self.duration_slider.valueChanged.connect(
            self.on_duration_slider_changed
        )

        # Max Duration Label
        max_duration_label = QLabel('10 s')
        max_duration_label.setFixedWidth(80)

        # Duration Layout
        duration_layout = QHBoxLayout(duration_widget)
        duration_layout.addWidget(self.cur_duration_label)
        duration_layout.addWidget(self.duration_slider)
        duration_layout.addWidget(max_duration_label)

        # Position Widget
        position_widget = QWidget()

        # Current Positions Label
        cur_position_label = QLabel('Current Position(m):  ')
        cur_position_label.setFixedWidth(300)

        # Current X Position
        self.cur_x_label = QLabel("0.0")
        self.cur_x_label.setMaximumSize(80, 20)

        # Current Y Position
        self.cur_y_label = QLabel("0.0")
        self.cur_y_label.setMaximumSize(80, 20)
        
        # Current Z Position
        self.cur_z_label = QLabel("0.0")
        self.cur_z_label.setMaximumSize(80, 20)
        
        # Goal Positions Label
        goal_position_label = QLabel('Goal Position(m):  ')
        goal_position_label.setFixedWidth(300)

        # Goal X Position
        self.goal_x_text = QPlainTextEdit()
        self.goal_x_text.setMaximumSize(80, 30)
        
        # Goal Y Position
        self.goal_y_text = QPlainTextEdit()
        self.goal_y_text.setMaximumSize(80, 30)
        
        # Positoin Layout
        position_layout = QHBoxLayout(position_widget)
        position_layout.addWidget(cur_position_label)
        position_layout.addWidget(self.cur_x_label)
        position_layout.addWidget(self.cur_y_label)
        position_layout.addWidget(self.cur_z_label)
        position_layout.addWidget(goal_position_label)
        position_layout.addWidget(self.goal_x_text)
        position_layout.addWidget(self.goal_y_text)

        # Play Widget
        play_widget = QWidget()

        # Start Button
        self.start_btn = QPushButton('Start')
        self.start_btn.clicked.connect(
            self.on_start_button_clicked
        )

        # Stop Button
        self.stop_btn = QPushButton('Stop')
        self.stop_btn.setDisabled(True)
        self.stop_btn.clicked.connect(
            self.on_stop_button_clicked
        )

        # Play Layout
        play_layout = QHBoxLayout(play_widget)
        play_layout.addWidget(self.start_btn)
        play_layout.addWidget(self.stop_btn)

        # Robot Controller Layout
        robot_controller_layout = QVBoxLayout(robot_controller)
        robot_controller_layout.addWidget(mode_widget)
        robot_controller_layout.addWidget(value_widget)
        robot_controller_layout.addWidget(duration_widget)
        robot_controller_layout.addWidget(position_widget)
        robot_controller_layout.addWidget(play_widget)
        
        # First Tab Layout
        first_tab_layout = QVBoxLayout(first_tab)
        first_tab_layout.addWidget(robot_monitor)
        first_tab_layout.addWidget(robot_controller)
        
        # TabWidget
        q_tab_widget : QTabWidget = self.findChild(QTabWidget, 'TabWidget')
        if q_tab_widget is None:
            self.node.get_logger().error('[RobotRQtWidget::initialize_ui] No such a name "TabWidget".')
            return
        q_tab_widget.addTab(first_tab, "Robot RQt Controller")

    def initialize_plot_widget(self, plot_widget):
        points_list = forward_kinematics(self.home_joint_positions, is_base=True, pose=self.home_pose)
        for idx, points in enumerate(points_list):
            self.lines[f'leg{idx+1}'] = gl.GLLinePlotItem(
                pos=points, color=(0.5, 0.5, 0.5, 1.0), width=5.0)
            self.scatters[f'leg{idx+1}'] = gl.GLScatterPlotItem(
                pos=points, size=10.0)
            plot_widget.addItem(self.lines[f'leg{idx+1}'])
            plot_widget.addItem(self.scatters[f'leg{idx+1}'])
        axis = get_base_axis(self.home_pose)
        self.axis['x'] = gl.GLLinePlotItem(pos=axis[0], color=(1.0, 0.0, 0.0, 1.0), width=5.0)
        self.axis['y'] = gl.GLLinePlotItem(pos=axis[1], color=(0.0, 1.0, 0.0, 1.0), width=5.0)
        self.axis['z'] = gl.GLLinePlotItem(pos=axis[2], color=(0.0, 0.0, 1.0, 1.0), width=5.0)
        plot_widget.addItem(self.axis['x'])
        plot_widget.addItem(self.axis['y'])
        plot_widget.addItem(self.axis['z'])

    def on_plotting_button_clicked(self):
        self.is_plotting = False if self.is_plotting else True
        
    def on_homing_button_clicked(self):
        self.mode = 'homing'
        self.value_slider.setDisabled(True)
        self.goal_x_text.setDisabled(True)
        self.goal_y_text.setDisabled(True)
        self.motion_params['goal_positions'] = self.home_joint_positions
    
    def on_spreading_button_clicked(self):
        self.mode = 'spreading'
        self.value_slider.setDisabled(False)
        self.goal_x_text.setDisabled(True)
        self.goal_y_text.setDisabled(True)
    
    def on_bouncing_button_clicked(self):
        self.mode = 'bouncing'
        self.value_slider.setDisabled(False)
        self.goal_x_text.setDisabled(True)
        self.goal_y_text.setDisabled(True)
    
    def on_walking_button_clicked(self):
        self.mode = 'walking'
        self.value_slider.setDisabled(False)
        self.goal_x_text.setDisabled(False)
        self.goal_y_text.setDisabled(False)

    def on_value_slider_changed(self, value):
        self.value[0] = value * 0.01
        self.cur_value_label.setText(f'Current Position:    {self.value[0]:.2f} m')
        self.motion_params['value'] = self.value

    def on_duration_slider_changed(self, value):
        self.duration = value * 0.2
        self.cur_duration_label.setText(f'Current Duration:    {self.duration:.1f} s')
        self.motion_params['duration'] = self.duration
    
    def on_start_button_clicked(self):
        if self.duration == 0:
            return
        if self.mode == 'walking':
            if self.goal_x != '0' and self.goal_y != '0':
                self.motion_params['goal_pose'][1, 0] = self.goal_x
                self.motion_params['goal_pose'][1, 1] = self.goal_y
                self.motion_params['goal_pose'][1, 2] = self.motion_params['current_pose'][1, 2]
        self.base_traj, self.foot_traj = self.functions[self.mode](self.motion_params)
       
        self.start_btn.setDisabled(True)
        self.stop_btn.setDisabled(False)

    def on_stop_button_clicked(self):
        self.base_traj = None
        self.foot_traj = None
        self.traj_idx = 0

        self.start_btn.setDisabled(False)
        self.stop_btn.setDisabled(True)

    def robot_state_callback(self, msg):
        self.motion_params['current_poes'][1, 0] = msg.pose.position.x
        self.motion_params['current_poes'][1, 1] = msg.pose.position.y
        self.motion_params['current_poes'][1, 2] = msg.pose.position.z
        self.motion_params['current_positions'] = np.array([x for x in msg.joint_state.position])
    
    def update_indicators(self):
        self.cur_x_label.setText(f"{self.motion_params['current_pose'][1, 0]:.3f}")
        self.cur_y_label.setText(f"{self.motion_params['current_pose'][1, 1]:.3f}")
        self.cur_z_label.setText(f"{self.motion_params['current_pose'][1, 2]:.3f}")
       
        self.goal_x = self.goal_x_text.toPlainText() 
        self.goal_y = self.goal_y_text.toPlainText() 

        if self.base_traj is not None:
            self.motion_params['current_pose'][1] = self.base_traj[self.traj_idx]
            self.motion_params['current_positions'] = self.foot_traj[self.traj_idx]
            self.traj_idx += 1
            
            if self.is_plotting:
                self.plot_robot()

            traj_msg = JointTrajectory()
            traj_msg.joint_names = ['j_11', 'j_12', 'j_13', 'j_21', 'j_22', 'j_23',
                                    'j_31', 'j_32', 'j_33', 'j_41', 'j_42', 'j_43',
                                    'j_51', 'j_52', 'j_53', 'j_61', 'j_62', 'j_63']
            traj_msg.points.append(JointTrajectoryPoint())
            traj_msg.points[0].positions = [pos for pos in self.motion_params['current_positions']]
            self.joint_trajectory_publisher.publish(traj_msg)

            if self.base_traj.shape[0] <= self.traj_idx:
                self.base_traj = None
                self.foot_traj = None
                self.traj_idx = 0
                
                self.start_btn.setDisabled(False)
                self.stop_btn.setDisabled(True)
    
    def plot_robot(self):
        points_list = forward_kinematics(self.motion_params['current_positions'],
                                         True,
                                         self.motion_params['current_pose'])
        for idx, points in enumerate(points_list):
            self.lines[f'leg{idx+1}'].setData(pos=points)
            self.scatters[f'leg{idx+1}'].setData(pos=points)
        axis = get_base_axis(self.motion_params['current_pose'])
        self.axis['x'].setData(pos=axis[0])
        self.axis['y'].setData(pos=axis[1])
        self.axis['z'].setData(pos=axis[2])

    def shutdown_widget(self):
        self.update_timer.stop()
        self.node.destroy_node()
