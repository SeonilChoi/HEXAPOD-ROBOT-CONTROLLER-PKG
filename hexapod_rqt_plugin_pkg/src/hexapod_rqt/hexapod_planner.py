import numpy as np

from hexapod_rqt.hexapod_kinematics import forward_kinematics, inverse_kinematics

def Linear(start: np.ndarray, end: np.ndarray, duration: float) -> np.ndarray:
    dt = 0.02
    theta = np.zeros((int(duration / dt), len(start)))
    for timestep in range(int(duration / dt)):
        t = (timestep + 1) * dt
        theta[timestep] = start + (end - start) * t / duration
    return theta

def LSPB(start: np.ndarray, end: np.ndarray, duration: float, rate: float) -> np.ndarray:
    vel = (2 * (end - start) / duration) * rate # Velocity
    nonzero_idx = np.where(np.abs(vel) > 1e-9)[0]

    tb = np.zeros(len(start)) # Start time of constant velocity phase
    tb[nonzero_idx] = (start[nonzero_idx] - end[nonzero_idx] + vel[nonzero_idx] * duration) / vel[nonzero_idx]
    
    acc = np.zeros(len(start)) # Acceleration
    acc[nonzero_idx] = np.power(vel[nonzero_idx], 2) / \
        (start[nonzero_idx] - end[nonzero_idx] + vel[nonzero_idx] * duration)
    
    constraint = np.zeros(len(start)) # Constraint on acceleration
    constraint[nonzero_idx] = 4 * (end[nonzero_idx] - start[nonzero_idx]) / duration**2
    if np.any(np.abs(acc[nonzero_idx]) < np.abs(constraint[nonzero_idx])):
        print("LSPB: Acceleration constraint violated")
        exit(1)

    dt = 0.02
    theta = np.zeros((int(duration / dt), len(start)))
    for timestep in range(int(duration / dt)):
        t = timestep * dt
        for i in range(len(start)):
            if t < tb[i]: # Acceleration phase
                theta[timestep, i] = start[i] + 0.5 * acc[i] * t**2
            elif t < duration - tb[i]: # Constant velocity phase
                theta[timestep, i] = start[i] + acc[i] * tb[i] * (t - tb[i] * 0.5)
            else: # Deceleration phase
                theta[timestep, i] = end[i] - 0.5 * acc[i] * (duration - t)**2
    return theta

def Polynomial_with_waypoint(start: np.ndarray, waypoint: np.ndarray, end: np.ndarray, duration: float, rate: float = 0.5) -> np.ndarray:
    f1 = duration * rate
    f2 = duration - f1

    theta = np.zeros((len(start), 8))
    theta[:, 0] = start
    theta[:, 2] = waypoint
    theta[:, 5] = waypoint
    theta[:, 6] = end

    A = np.zeros((len(start), 8, 8))
    A[:, 0, 0] = 1
    A[:, 1, 1] = 1
    A[:, 2, 0] = 1
    A[:, 2, 1] = f1
    A[:, 2, 2] = f1**2
    A[:, 2, 3] = f1**3
    A[:, 3, 2] = 2 * f1
    A[:, 3, 3] = 3 * f1**2
    A[:, 3, 5] = -1
    A[:, 4, 2] = 2
    A[:, 4, 3] = 6 * f1
    A[:, 4, 6] = -2
    A[:, 5, 4] = 1
    A[:, 6, 4] = 1
    A[:, 6, 5] = f2
    A[:, 6, 6] = f2**2
    A[:, 6, 7] = f2**3
    A[:, 7, 5] = 1
    A[:, 7, 6] = 2 * f2
    A[:, 7, 7] = 3 * f2**2

    b = np.zeros((len(start), 8))
    for i in range(len(start)):
        b[i] = np.linalg.inv(A[i]) @ theta[i]
    
    dt = 0.02
    theta = np.zeros((int(duration / dt), len(start)))
    for timestep in range(int(duration / dt)):
        t = (timestep + 1) * dt
        for i in range(len(start)):
            if t <= f1:
                theta[timestep, i] = b[i, 0] + b[i, 1] * t + b[i, 2] * t**2 + b[i, 3] * t**3
            else:
                theta[timestep, i] = b[i, 4] + b[i, 5] * (t - f1) + \
                    b[i, 6] * (t - f1)**2 + b[i, 7] * (t - f1)**3
    return theta

def stepping(pre_position, pre_base_pose, value, duration, step_length, 
            offset_value, offset_duration, offset_length,
            swing_foot_list, support_foot_list, is_periodic):
    value = value * 2 if is_periodic else value
    offset_value = (offset_duration / duration) * value / 2

    # Base
    traj_base_pose = np.zeros((step_length, 3)) + pre_base_pose
    one_step_base_distance = value / 2 / step_length
    for i in range(1, step_length + 1):
        traj_base_pose[i-1] += one_step_base_distance * i

    # Support Foot
    start_position_sup_1 = pre_position.copy()
    start_pose_sup_1 = forward_kinematics(start_position_sup_1, is_world=True)

    end_pose_sup_1 = start_pose_sup_1.copy()
    end_pose_sup_1 -= value/4
    end_pose_sup_1[:, 2] -= 0.07
    end_position_sup_1 = inverse_kinematics(end_pose_sup_1, is_world=True)
    
    start_position_sup_2 = end_position_sup_1.copy()
    start_pose_sup_2 = forward_kinematics(start_position_sup_2, is_world=True)

    end_pose_sup_2 = start_pose_sup_2.copy()
    end_pose_sup_2 -= value/4
    end_pose_sup_2[:, 2] += 0.07
    end_position_sup_2 = inverse_kinematics(end_pose_sup_2, is_world=True)

    traj_sup_1 = Linear(start_position_sup_1, end_position_sup_1, duration/2)
    traj_sup_2 = Linear(start_position_sup_2, end_position_sup_2, duration/2)
    traj_sup = np.concatenate((traj_sup_1, traj_sup_2), axis=0)

    # Swing Foot 1
    start_position_leg_1_1 = pre_position.copy()
    start_pose_leg_1_1 = forward_kinematics(start_position_leg_1_1, is_world=True)

    way_pose_leg_1_1 = start_pose_leg_1_1.copy()
    way_pose_leg_1_1 += value/4
    way_pose_leg_1_1[:, 2] += 0.1
    way_position_leg_1_1 = inverse_kinematics(way_pose_leg_1_1, is_world=True)
  
    end_pose_leg_1_1 = way_pose_leg_1_1.copy()
    end_pose_leg_1_1 += value/4 + offset_value*2
    end_pose_leg_1_1[:, 2] = forward_kinematics(
        traj_sup[step_length-offset_length*2 - 1], is_world=True)[:, 2]
    end_position_leg_1_1 = inverse_kinematics(end_pose_leg_1_1, is_world=True)

    start_pose_leg_1_2 = end_pose_leg_1_1.copy()
    start_position_leg_1_2 = end_position_leg_1_1.copy()
  
    end_pose_leg_1_2 = start_pose_leg_1_2.copy()
    end_pose_leg_1_2 -= offset_value*2
    end_pose_leg_1_2[:, 2] = forward_kinematics(
            traj_sup[-1], is_world=True)[:, 2]
    end_position_leg_1_2 = inverse_kinematics(
        end_pose_leg_1_2, is_world=True)

    traj_leg_1_1 = Polynomial_with_waypoint(
        start_position_leg_1_1, way_position_leg_1_1, end_position_leg_1_1, duration-offset_duration*2)
    traj_leg_1_2 = Linear(
        start_position_leg_1_2, end_position_leg_1_2, offset_duration*2)
    traj_leg_1 = np.concatenate((traj_leg_1_1, traj_leg_1_2), axis=0)
    
    # Swing Foot 2
    start_position_leg_2_1 = pre_position.copy()
    start_pose_leg_2_1 = forward_kinematics(start_position_leg_2_1, is_world=True)

    end_pose_leg_2_1 = start_pose_leg_2_1.copy()
    end_pose_leg_2_1 -= offset_value
    end_pose_leg_2_1[:, 2] = forward_kinematics(
        traj_sup[offset_length-1], is_world=True)[:, 2]
    end_position_leg_2_1 = inverse_kinematics(
        end_pose_leg_2_1, is_world=True)

    start_pose_leg_2_2 = end_pose_leg_2_1.copy()
    start_position_leg_2_2 = end_position_leg_2_1.copy()

    way_pose_leg_2_2 = start_pose_leg_2_2.copy()
    way_pose_leg_2_2 += value/4
    way_pose_leg_2_2[:, 2] += 0.1
    way_position_leg_2_2 = inverse_kinematics(
        way_pose_leg_2_2, is_world=True)

    end_pose_leg_2_2 = way_pose_leg_2_2.copy()
    end_pose_leg_2_2 += value/4 + offset_value*2
    end_pose_leg_2_2[:, 2] = forward_kinematics(
        traj_sup[step_length-offset_length-1], is_world=True)[:, 2]
    end_position_leg_2_2 = inverse_kinematics(
        end_pose_leg_2_2, is_world=True)

    start_pose_leg_2_3 = end_pose_leg_2_2.copy()
    start_position_leg_2_3 = end_position_leg_2_2.copy()

    end_pose_leg_2_3 = start_pose_leg_2_3.copy()
    end_pose_leg_2_3 -= offset_value
    end_pose_leg_2_3[:, 2] = forward_kinematics(
        traj_sup[-1], is_world=True)[:, 2]
    end_position_leg_2_3 = inverse_kinematics(
        end_pose_leg_2_3, is_world=True)

    traj_leg_2_1 = Linear(start_position_leg_2_1, end_position_leg_2_1, offset_duration)
    traj_leg_2_2 = Polynomial_with_waypoint(
        start_position_leg_2_2, way_position_leg_2_2, end_position_leg_2_2, duration-offset_duration*2)
    traj_leg_2_3 = Linear(start_position_leg_2_3, end_position_leg_2_3, offset_duration)
    traj_leg_2 = np.concatenate((traj_leg_2_1, traj_leg_2_2, traj_leg_2_3), axis=0)

    # Swing Foot 3
    start_position_leg_3_1 = pre_position.copy()
    start_pose_leg_3_1 = forward_kinematics(start_position_leg_3_1, is_world=True)

    end_pose_leg_3_1 = start_pose_leg_3_1.copy()
    end_pose_leg_3_1 -= offset_value*2
    end_pose_leg_3_1[:, 2] = forward_kinematics(
        traj_sup[offset_length*2-1], is_world=True)[:, 2]
    end_position_leg_3_1 = inverse_kinematics(
        end_pose_leg_3_1, is_world=True)

    start_pose_leg_3_2 = end_pose_leg_3_1.copy()
    start_position_leg_3_2 = end_position_leg_3_1.copy()

    way_pose_leg_3_2 = start_pose_leg_3_2.copy()
    way_pose_leg_3_2 += value/4
    way_pose_leg_3_2[:, 2] += 0.1
    way_position_leg_3_2 = inverse_kinematics(
        way_pose_leg_3_2, is_world=True)

    end_pose_leg_3_2 = way_pose_leg_3_2.copy()
    end_pose_leg_3_2 += value/4 + offset_value*2
    end_pose_leg_3_2[:, 2] = forward_kinematics(
        traj_sup[-1], is_world=True)[:, 2]
    end_position_leg_3_2 = inverse_kinematics(end_pose_leg_3_2, is_world=True)

    traj_leg_3_1 = Linear(start_position_leg_3_1, end_position_leg_3_1, offset_duration*2)
    traj_leg_3_2 = Polynomial_with_waypoint(
        start_position_leg_3_2, way_position_leg_3_2, end_position_leg_3_2, duration-offset_duration*2)
    traj_leg_3 = np.concatenate((traj_leg_3_1, traj_leg_3_2), axis=0)

    # Total
    total_traj = np.zeros((step_length, 18))

    swing_leg_1 = [swing_foot_list[0], swing_foot_list[0]+6, swing_foot_list[0]+12]
    swing_leg_2 = [swing_foot_list[1], swing_foot_list[1]+6, swing_foot_list[1]+12]
    swing_leg_3 = [swing_foot_list[2], swing_foot_list[2]+6, swing_foot_list[2]+12]
    support_leg_1 = [support_foot_list[0], support_foot_list[0]+6, support_foot_list[0]+12]
    support_leg_2 = [support_foot_list[1], support_foot_list[1]+6, support_foot_list[1]+12]
    support_leg_3 = [support_foot_list[2], support_foot_list[2]+6, support_foot_list[2]+12]

    total_traj[:, swing_leg_1] = traj_leg_1[:, swing_leg_1]
    total_traj[:, swing_leg_2] = traj_leg_2[:, swing_leg_2]
    total_traj[:, swing_leg_3] = traj_leg_3[:, swing_leg_3]
    total_traj[:, support_leg_1] = traj_sup[:, support_leg_1]
    total_traj[:, support_leg_2] = traj_sup[:, support_leg_2]
    total_traj[:, support_leg_3] = traj_sup[:, support_leg_3]

    total_pose = []
    for pos in total_traj:
        total_pose.append(forward_kinematics(pos, is_world=True))
    traj_base_pose[:, -1] = np.array(total_pose)[:, support_foot_list[0], -1]

    return total_traj, total_traj[-1], traj_base_pose, traj_base_pose[-1]
    
def homing(pre_position, goal_position, current_pose, duration):
    start_position_1 = pre_position.copy()
    start_pose_1 = forward_kinematics(start_position_1, is_world=False)

    leg_list_1 = [0, 2, 4]
    leg_list_1_1 = [0, 2, 4, 6, 8, 10, 12, 14, 16]
    way_pose_1 = start_pose_1.copy()
    way_pose_1[leg_list_1, 2] += 0.1
    way_position_1 = inverse_kinematics(way_pose_1, is_world=False)

    end_position_1 = way_position_1.copy()
    end_position_1[leg_list_1_1] = goal_position[leg_list_1_1]

    start_position_2 = end_position_1.copy()
    start_pose_2 = forward_kinematics(start_position_2, is_world=False)

    leg_list_2 = [1, 3, 5]
    leg_list_2_1 = [1, 3, 5, 7, 9, 11, 13, 15, 17]
    way_pose_2 = start_pose_2.copy()
    way_pose_2[leg_list_2, 2] += 0.1
    way_position_2 = inverse_kinematics(way_pose_2, is_world=False)

    end_position_2 = way_position_2.copy()
    end_position_2[leg_list_2_1] = goal_position[leg_list_2_1]

    traj_1 = LSPB(start_position_1, way_position_1, duration, 0.7)
    traj_2 = np.zeros((50, 18)) + way_position_1
    traj_3 = LSPB(way_position_1, end_position_1, duration, 0.7)
    traj_4 = np.zeros((50, 18)) + end_position_1
    traj_5 = LSPB(start_position_2, way_position_2, duration, 0.7)
    traj_6 = np.zeros((50, 18)) + way_position_2
    traj_7 = LSPB(way_position_2, end_position_2, duration, 0.7)
    traj = np.concatenate((traj_1, traj_2, traj_3, traj_4, traj_5, traj_6, traj_7), axis=0)
    
    traj_pose = []
    for j in traj:
        traj_pose.append(forward_kinematics(j, is_world=True))

    half_index = int(traj.shape[0] / 2)
    base_traj = np.zeros((traj.shape[0], 3)) + current_pose
    base_traj[:half_index, -1] = np.array(traj_pose)[:half_index, 1, -1]
    base_traj[half_index:, -1] = np.array(traj_pose)[half_index:, 0, -1]
    return traj, base_traj


def moving(pre_position, x, duration):
    start_position_1 = pre_position.copy()
    start_pose_1 = forward_kinematics(start_position_1, is_world=False)

    leg_list_1 = [0, 2, 4]
    way_pose_1 = start_pose_1.copy()
    way_pose_1[leg_list_1, 2] += 0.05
    way_position_1 = inverse_kinematics(way_pose_1, is_world=False)

    end_pose_1 = start_pose_1.copy()
    end_pose_1[leg_list_1, 0] += x
    end_position_1 = inverse_kinematics(end_pose_1, is_world=False)
    
    start_position_2 = end_position_1.copy()
    start_pose_2 = forward_kinematics(start_position_2, is_world=False)

    leg_list_2 = [1, 3, 5]
    way_pose_2 = start_pose_2.copy()
    way_pose_2[leg_list_2, 2] += 0.05
    way_position_2 = inverse_kinematics(way_pose_2, is_world=False)

    end_pose_2 = start_pose_2.copy()
    end_pose_2[leg_list_2, 0] += x
    end_position_2 = inverse_kinematics(end_pose_2, is_world=False)

    traj_1 = Polynomial_with_waypoint(start_position_1, way_position_1, end_position_1, duration)
    traj_2 = Polynomial_with_waypoint(start_position_2, way_position_2, end_position_2, duration)
    return np.concatenate((traj_1, traj_2), axis=0)

def walking(pre_position, current_pose, goal_pose, stride, duration):
    offset_length = int(duration / 0.02 * 0.2)
    offset_duration = offset_length * 0.02

    distance = np.linalg.norm(goal_pose - current_pose)
    direction = (goal_pose - current_pose) / distance
    steps = int(distance / stride) + 1 # for the last step
    step_length = int(duration / 0.02)
    total_length = step_length * steps

    real_goal_pose = (steps - 1) * stride * direction
 
    gait_trajectory = np.zeros((total_length, 18))
    base_trajectory = np.zeros((total_length, 3))

    value = direction * stride
    offset_value = (offset_duration / duration) * direction * stride

    leg_list_right = [0, 2, 4]
    leg_list_left = [1, 3, 5]
    leg_list = [leg_list_right, leg_list_left]

    end_position = pre_position.copy()
    end_pose = current_pose.copy()
    for step in range(steps):
        start_index = step * step_length
        end_index = start_index + step_length

        swing_foot_list = leg_list[step % 2]
        support_foot_list = leg_list[(step + 1) % 2]

        swing_foot_list = np.random.permutation(swing_foot_list)
        support_foot_list = np.random.permutation(support_foot_list)

        is_periodic = False if step == 0 or step==steps-1 else True

        gait_trajectory[start_index:end_index, :], end_position, base_trajectory[start_index:end_index, :], end_pose = stepping(
            end_position, end_pose, value, duration, step_length, offset_value, offset_duration, offset_length, swing_foot_list, support_foot_list, is_periodic)
    return gait_trajectory, base_trajectory, real_goal_pose
