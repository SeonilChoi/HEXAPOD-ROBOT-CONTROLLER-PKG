import numpy as np
from abc import ABC, abstractmethod

class Kinematics(ABC):
    def __init__(self, dh_params):
        self.dh_params = dh_params

    def get_world_to_base(self, orientation: np.ndarray, position: np.ndarray):
        R, P, Y = orientation
        x, y, z = position

        Rz = np.array([[np.cos(Y), -np.sin(Y), 0],
                       [np.sin(Y),  np.cos(Y), 0],
                       [        0,          0, 1]])
    
        Ry = np.array([[ np.cos(P), 0, np.sin(P)],
                       [        0,  1,         0],
                       [-np.sin(P), 0, np.cos(P)]])
    
        Rx = np.array([[1,         0,          0],
                       [0, np.cos(R), -np.sin(R)],
                       [0, np.sin(R),  np.cos(R)]])

        R = Rz @ Ry @ Rx
        p = np.array([x, y, z]).reshape(3, 1)
        return np.r_[np.c_[R, p], [[0, 0, 0, 1]]]

    def get_base_to_world(self, pose: np.ndarray):
        Twb = self.get_world_to_base(pose[0], pose[1])
        R = Twb[:3, :3]
        p = Twb[:3, -1]
        return np.r_[np.c_[R.T, -R.T@p], [[0, 0, 0, 1]]]

    def get_base_axis(self, pose: np.ndarray):
        T = self.get_world_to_base(pose[0], pose[1])
        axis = np.r_[np.eye(3)*0.25, [[1, 1, 1]]]
        base_axis = np.dot(T, axis)[:3].T
        origin = np.array([pose[1], pose[1], pose[1]])
        return np.concatenate((origin, base_axis), axis=1).reshape(3, 2, 3)
        
    def get_homogeneous_transformation_matrix(self, params: np.ndarray):
        a, al, d, th = params
        return np.array([[           np.cos(th),           -np.sin(th),           0,             a],
                         [np.sin(th)*np.cos(al), np.cos(th)*np.cos(al), -np.sin(al), -np.sin(al)*d],
                         [np.sin(th)*np.sin(al), np.cos(th)*np.sin(al),  np.cos(al),  np.cos(al)*d],
                         [                    0,                     0,           0,             1]])

    @abstractmethod
    def update_dh_params(self, theta_list: np.ndarray):
        pass

    @abstractmethod
    def forward_kinematics(self, pose: np.ndarray, is_world: bool):
        pass
    
    @abstractmethod
    def inverse_kinematics(self, point: np.ndarray):
        pass

class ThreeDoFSpatialManipulator(Kinematics):
    def __init__(self, dh_params, link_list):
        super().__init__(dh_params)
        self.l0 = link_list[0]
        self.l1 = link_list[1]
        self.l2 = link_list[2]

    def update_dh_params(self, theta_list):
        for i, th in enumerate(theta_list):
            self.dh_params[i, -1] = th

    def forward_kinematics(self, pose: np.ndarray, is_world: bool):
        T_list = []
        p_list = []
        T = self.get_world_to_base(pose[0], pose[1])
        for params in self.dh_params:
            T = np.dot(T, self.get_homogeneous_transformation_matrix(params))
            T_list.append(T)
            p_list.append(T[:3, -1].reshape(3))
        return np.array(T_list), np.array(p_list)

    def inverse_kinematics(self, point: np.ndarray):
        x, y, z = point

        # Theta 1
        th1 = np.arctan2(y, x)

        # Theta 3
        Px = x / np.cos(th1)
        Pz = z - self.l0
        c3 = (Px**2 + Pz**2 - self.l1**2 - self.l2**2) / (2 * self.l1 * self.l2)
        s3 = np.sqrt(1 - c3**2)
        th3 = np.arctan2(s3, c3)

        # Theta 2
        m = self.l1 + self.l2 * np.cos(th3)
        n = self.l2 * np.sin(th3)
        th2 = np.arctan2(Pz, Px) - np.arctan2(n, m)

        return np.array([th1, th2, th3])
    
class HexapodOneLeg(Kinematics):
    def __init__(self, dh_params, link_list, leg_num):
        super().__init__(dh_params)
        self.l0 = link_list[0]
        self.l1 = link_list[1]
        self.l2 = link_list[2]
        self.l3 = link_list[3]
        self.leg_num = leg_num # start 1

    def update_dh_params(self, theta_list):
        theta_index = [1, 3, 4]
        for idx, th in zip(theta_index, theta_list):
            self.dh_params[idx, -1] = th
        self.dh_params[3, -1] = self.dh_params[3, -1] + np.pi/2 if self.leg_num <= 3 else self.dh_params[3, -1] - np.pi/2

    def forward_kinematics(self, pose: np.ndarray, is_world: bool):
        T_list = []
        p_list = []
        T = self.get_world_to_base(pose[0], pose[1]) if is_world else np.eye(4) 
        for i, params in enumerate(self.dh_params):
            if i == 0 and is_world is False:
                continue
            T = np.dot(T, self.get_homogeneous_transformation_matrix(params))
            T_list.append(T)
            p_list.append(T[:3, -1].reshape(3))
        return np.array(T_list), np.array(p_list)

    def inverse_kinematics(self, point: np.ndarray):
        x, y, z = point
        x -= self.l0

        # Theta 1
        th1 = np.arctan2(y, x)

        # Theta 3
        Px = x / np.cos(th1)
        Pz = z - self.l1
        c3 = (Px**2 + Pz**2 - self.l2**2 - self.l3**2) / (2 * self.l2 * self.l3)
        c3 = np.clip(c3, -1.0, 1.0)
        s3 = np.sqrt(1 - c3**2)
        th3 = np.arctan2(s3, c3)
        th3 = -th3 if self.leg_num <= 3 else th3

        # Theta 2
        m = self.l2 + self.l3 * np.cos(th3)
        n = self.l3 * np.sin(th3) if self.leg_num <= 3 else self.l3 * np.sin(-th3)
        th2 = np.arctan2(Pz, Px) - np.arctan2(n, m)
        th2 = th2 - np.pi/2 if self.leg_num <= 3 else -th2 + np.pi/2

        return np.array([th1, th2, th3])
    
class HexapodRobot:
    def __init__(self, dh_params_list, link_list):
        self.robot = {}
        self.initialize(dh_params_list, link_list)

    def initialize(self, dh_params_list: np.ndarray, link_list: np.ndarray):
        for i, dh_params in enumerate(dh_params_list):
            self.robot[f'leg{i+1}'] = HexapodOneLeg(dh_params, link_list, i+1)

    def update_dh_params_list(self, theta_lists: np.ndarray):
        for i, theta_list in enumerate(theta_lists):
            self.robot[f'leg{i+1}'].update_dh_params(theta_list)

    def forward_kinematics(self, pose: np.ndarray, is_world: bool = True):
        positions = []
        for leg in self.robot.values():
            positions.append(leg.forward_kinematics(pose, is_world)[1])
        return np.array(positions)
    
    def inverse_kinematics(self, point_list: np.ndarray):
        theta_lists = []
        for i, point in enumerate(point_list):
            theta_lists.append(self.robot[f'leg{i+1}'].inverse_kinematics(point))
        return np.array(theta_lists)

def load_hexapod_robot():
    l0, l1, l2, l3 = 0.35, 0.25, 0.7, 1.2
    
    def get_dh_params(index):
        if index < 3:
            return np.array([
                [ 0,       0,  0, -np.deg2rad(30)-np.deg2rad(60)*index],
                [l0,       0,  0,                                    0],
                [ 0,       0, l1,                                    0], 
                [ 0, np.pi/2,  0,                              np.pi/2],
                [l2,       0,  0,                                    0], 
                [l3,       0,  0,                                    0]
            ])
        else:
            return np.array([
                [ 0,        0,  0, np.deg2rad(30)+np.deg2rad(60)*(5-index)],
                [l0,        0,  0,                                       0],
                [ 0,        0, l1,                                       0], 
                [ 0, -np.pi/2,  0,                                -np.pi/2],
                [l2,        0,  0,                                       0], 
                [l3,        0,  0,                                       0]
            ])
        
    dh_params_list = np.array([get_dh_params(i) for i in range(6)])
    link_list = np.array([l0, l1, l2, l3])
    return HexapodRobot(dh_params_list, link_list)