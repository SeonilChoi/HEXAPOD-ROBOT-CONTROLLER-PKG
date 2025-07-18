import numpy as np

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
                theta[timestep, i] = b[i, 4] + b[i, 5] * (t - f1) + b[i, 6] * (t - f1)**2 + b[i, 7] * (t - f1)**3
    return theta