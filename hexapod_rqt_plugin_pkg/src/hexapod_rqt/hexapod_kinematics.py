import numpy as np

# Forward Kinematics
def get_dh_params(leg_idx, is_world):
    l0, l1, l2, l3 = 0.32, 0.25, 0.7, 1.2
    dh_params = None
    if leg_idx < 3:
        dh_params = np.array([
            [ 0,       0,  0, -np.deg2rad(30)-np.deg2rad(60)*leg_idx],
            [l0,       0,  0,                                    0],
            [ 0,       0, l1,                                    0], 
            [ 0, np.pi/2,  0,                              np.pi/2],
            [l2,       0,  0,                                    0], 
            [l3,       0,  0,                                    0]
        ])
    else:
        dh_params = np.array([
            [ 0,        0,  0, np.deg2rad(30)+np.deg2rad(60)*(5-leg_idx)],
            [l0,        0,  0,                                       0],
            [ 0,        0, l1,                                       0], 
            [ 0, -np.pi/2,  0,                                -np.pi/2],
            [l2,        0,  0,                                       0], 
            [l3,        0,  0,                                       0]
        ])
    return dh_params if is_world else dh_params[1:]

def get_homogeneous_transformation_matrix(params):
    a, al, d, th = params
    return np.array([[          np.cos(th),           -np.sin(th),           0,             a],
                    [np.sin(th)*np.cos(al), np.cos(th)*np.cos(al), -np.sin(al), -np.sin(al)*d],
                    [np.sin(th)*np.sin(al), np.cos(th)*np.sin(al),  np.cos(al),  np.cos(al)*d],
                    [                    0,                     0,           0,             1]])

def get_base_to_end_effector(dh_params):
    T = np.eye(4)
    T_list = []
   
    for params in dh_params:
        T = np.dot(T, get_homogeneous_transformation_matrix(params))
        T_list.append(T)
    return np.array(T_list)

def update_dh_params(dh_params, joint_pos, leg_idx, is_world):
    if is_world:
        dh_params[1, 3] += joint_pos[leg_idx]
        dh_params[3, 3] += joint_pos[leg_idx + 6]
        dh_params[4, 3] += joint_pos[leg_idx + 12]
    else:
        dh_params[0, 3] += joint_pos[leg_idx]
        dh_params[2, 3] += joint_pos[leg_idx + 6]
        dh_params[3, 3] += joint_pos[leg_idx + 12]
    return dh_params

def forward_kinematics(joint_pos, is_world):
    pos = np.zeros((6, 3))
    for leg_idx in range(6):
        dh_params = get_dh_params(leg_idx, is_world)
        dh_params = update_dh_params(dh_params, joint_pos, leg_idx, is_world)
        T = get_base_to_end_effector(dh_params)
        pos[leg_idx] = T[-1, :3, 3]
    return pos

def get_base_to_end_effector_2(pose, dh_params):
    T = get_world_to_base(pose[0], pose[1]) 
    T_list = []
   
    for params in dh_params:
        T = np.dot(T, get_homogeneous_transformation_matrix(params))
        T_list.append(T)
    return np.array(T_list)

def forward_kinematics_2(pose, joint_pos, is_world):
    pos = np.zeros((6, 6, 3))
    for leg_idx in range(6):
        dh_params = get_dh_params(leg_idx, is_world)
        dh_params = update_dh_params(dh_params, joint_pos, leg_idx, is_world)
        T = get_base_to_end_effector_2(pose, dh_params)
        pos[leg_idx] = T[:, :3, -1].reshape(6, 3)
    return pos

# Inverse Kinematics
def get_leg_based_point(leg_idx, point, is_world):
    end_effector_pos = np.array([point[0], point[1], point[2], 1])
    dh_params = get_dh_params(leg_idx, is_world)
    T = get_base_to_end_effector(dh_params)
    T01 = T[0]
    R01 = T01[:3, :3]
    p01 = T01[:3, -1].reshape(3, 1)
    T10 = np.r_[np.c_[R01.T, -R01.T@p01], [[0, 0, 0, 1]]]
    leg_based_point = T10 @ end_effector_pos
    return leg_based_point[0], leg_based_point[1], leg_based_point[2]

def inverse_kinematics(points, is_world):
    l0, l1, l2, l3 = 0.32, 0.25, 0.7, 1.2
    joint_pos = np.zeros((18,))
    for leg_idx in range(6):
        x, y, z = get_leg_based_point(leg_idx, points[leg_idx], is_world) if is_world else points[leg_idx]
        x -= l0

        # Theta 1
        th1 = np.arctan2(y, x)

        # Theta 3
        Px = x / np.cos(th1)
        Pz = z - l1
        c3 = (Px**2 + Pz**2 - l2**2 - l3**2) / (2 * l2 * l3)
        c3 = np.clip(c3, -1.0, 1.0)
        s3 = np.sqrt(1 - c3**2)
        th3 = np.arctan2(s3, c3)
        th3 = -th3 if leg_idx < 3 else th3

        # Theta 2
        m = l2 + l3 * np.cos(th3)
        n = l3 * np.sin(th3) if leg_idx < 3 else l3 * np.sin(-th3)
        th2 = np.arctan2(Pz, Px) - np.arctan2(n, m)
        th2 = th2 - np.pi/2 if leg_idx < 3 else -th2 + np.pi/2

        joint_pos[leg_idx] = th1
        joint_pos[leg_idx + 6] = th2
        joint_pos[leg_idx + 12] = th3
    return joint_pos

def get_world_to_base(orientation, position):
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

def get_base_to_world(pose):
    Twb = get_world_to_base(pose[0], pose[1])
    R = Twb[:3, :3]
    p = Twb[:3, -1]
    return np.r_[np.c_[R.T, -R.T@p], [[0, 0, 0, 1]]]

def get_base_axis(pose):
    T = get_world_to_base(pose[0], pose[1])
    axis = np.r_[np.eye(3)*0.25, [[1, 1, 1]]]
    base_axis = np.dot(T, axis)[:3].T
    origin = np.array([pose[1], pose[1], pose[1]])
    return np.concatenate((origin, base_axis), axis=1).reshape(3, 2, 3)
