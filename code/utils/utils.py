import numpy as np
import pandas as pd
import math
import torch
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy import signal
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

class HumanMotion(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __getitem__(self, item):
        return self.x[item].astype(float), self.y[item].astype(float)

    def __len__(self):
        return self.x.shape[0]


def generate_dataset(x, y, batch_size = 128, shuffle = True):
    x_train, x_eval, y_train, y_eval = train_test_split(x, y, test_size=0.4, shuffle=False)
    x_valid, x_test, y_valid, y_test = train_test_split(x_eval, y_eval, test_size=0.5,
                                                        shuffle=False)
    train_loader = DataLoader(HumanMotion(x_train, y_train), num_workers=8,
                              batch_size=batch_size, shuffle=shuffle, drop_last=True)
    valid_loader = DataLoader(HumanMotion(x_valid, y_valid), num_workers=8,
                             batch_size=batch_size, shuffle=shuffle, drop_last=False)
    test_loader = DataLoader(HumanMotion(x_test, y_test), num_workers=8,
                              batch_size=batch_size, shuffle=shuffle, drop_last=False)
    return train_loader, valid_loader, test_loader, x_test, y_test

# Code based on:
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

# Expects tuples of (state, next_state, action, reward, done)
class ReplayBuffer(object):
    '''
    Change the buffer to array and delete for loop.
    '''
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def get(self, idx):
        return self.storage[idx]

    def add(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def add_final_reward(self, final_reward, steps, delay=0):
        len_buffer = len(self.storage)
        for i in range(len_buffer - steps - delay, len_buffer - delay):
            item = list(self.storage[i])
            item[3] += final_reward
            self.storage[i] = tuple(item)

    def add_specific_reward(self, reward_vec, idx_vec):
        for i in range(len(idx_vec)):
            time_step_num = int(idx_vec[i])
            item = list(self.storage[time_step_num])
            item[3] += reward_vec[i]
            self.storage[time_step_num] = tuple(item)

    def sample_on_policy(self, batch_size, option_buffer_size):
        return self.sample_from_storage(batch_size, self.storage[-option_buffer_size:])

    def sample(self, batch_size):
        return self.sample_from_storage(batch_size, self.storage)

    @staticmethod
    def sample_from_storage(batch_size, storage):
        ind = np.random.randint(0, len(storage), size=batch_size)
        x, y, u, r, d, p = [], [], [], [], [], []
        for i in ind:
            X, Y, U, R, D, P = storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))
            p.append(np.array(P, copy=False))
        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), \
               np.array(d).reshape(-1, 1), np.array(p).reshape(-1, 1)


# Expects tuples of (state, next_state, action, reward, done)
class ReplayBufferMat(object):
    '''
    Change the buffer to array and delete for loop.
    '''
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0
        self.data_size = 0

    def add(self, data):
        data = list(data)
        if 0 == len(self.storage):
            for item in data:
                self.storage.append(np.asarray(item).reshape((1, -1)))
        else:
            if self.storage[0].shape[0] < int(self.max_size):
                for i in range(len(data)):
                    self.storage[i] = np.r_[self.storage[i], np.asarray(data[i]).reshape((1, -1))]
            else:
                for i in range(len(data)):
                    self.storage[i][int(self.ptr)] = np.asarray(data[i]).reshape((1, -1))
                self.ptr = (self.ptr + 1) % self.max_size
        self.data_size = len(self.storage[0])

    def sample_on_policy(self, batch_size, option_buffer_size):
        return self.sample_from_storage(
            batch_size, start_idx = self.storage[0].shape[0] - option_buffer_size)

    def sample(self, batch_size):
        return self.sample_from_storage(batch_size)

    def sample_from_storage(self, batch_size, start_idx = 0):
        buffer_len = self.storage[0].shape[0]
        ind = np.random.randint(start_idx, buffer_len, size=batch_size)
        data_list = []
        # if buffer_len > 9998:
        #     print(buffer_len, ind)
        for i in range(len(self.storage)):
            # if buffer_len > 9998:
            #     print('{},shape:{}'.format(i, self.storage[i].shape))
            data_list.append(self.storage[i][ind])
        return tuple(data_list)

    def add_final_reward(self, final_reward, steps):
        self.storage[3][-steps:] += final_reward

def calc_array_symmetry(array_a, array_b):
    cols = array_a.shape[-1]
    dist = np.zeros(cols)
    for c in range(cols):
        dist[c] = 1 - distance.cosine(array_a[:, c], array_b[:, c])
    return np.mean(dist)


def calc_cos_similarity(joint_angle_resample, human_joint_angle):
    joint_num = human_joint_angle.shape[0]
    dist = np.zeros(joint_num)
    for c in range(joint_num):
        dist[c] = 1 - distance.cosine(joint_angle_resample[c, :], human_joint_angle[c, :])
    return np.mean(dist)


def calc_cross_gait_reward(gait_state_mat, gait_velocity, reward_name):
    """
    reward_name_vec =['r_d', 'r_s', 'r_f', 'r_n', 'r_gv', 'r_lhs', 'r_gs', 'r_cg', 'r_fr', 'r_po']
    """
    cross_gait_reward = 0.0
    reward_str_list = []
    frame_num = gait_state_mat.shape[0]
    joint_deg_mat = joint_state_to_deg(gait_state_mat[:, :-2])
    ankle_to_hip_deg_mat = joint_deg_mat[:, [0, 3]] - joint_deg_mat[:, [1, 4]]
    if 'r_gv' in reward_name:
        '''
        gait velocity
        '''
        reward_str_list.append('r_gv')
        cross_gait_reward += 0.2 * np.mean(gait_velocity)

    if 'r_lhs' in reward_name:
        '''
        0: left heel strike: the left foot should contact ground between 40% to 60% gait cycle
        Theoretical situation: 0, -1: right foot strike; 50: left foot strike
        '''
        reward_str_list.append('r_lhs')

        l_foot_contact_vec = signal.medfilt(gait_state_mat[:, -1], 3)
        l_foot_contact_vec[1:] -= l_foot_contact_vec[:-1]
        l_foot_contact_vec[0] = 0
        if 0 == np.mean(l_foot_contact_vec == 1):
            # print(gait_state_mat_sampled)
            return cross_gait_reward, reward_str_list
        l_heel_strike_idx = np.where(l_foot_contact_vec == 1)[0][0]
        cross_gait_reward += 0.2 * (1.0 - np.tanh((l_heel_strike_idx / (frame_num + 0.0) - 0.5) ** 2))


        if 'r_gs' in reward_name:
            '''
            1: gait symmetry
            '''
            reward_str_list.append('r_gs')

            r_gait_state_origin = gait_state_mat[:, np.r_[0:3, -2]]
            l_gait_state_origin = gait_state_mat[:, np.r_[3:6, -1]]
            l_gait_state = np.zeros(l_gait_state_origin.shape)
            l_gait_state[0:(frame_num - l_heel_strike_idx), :] = l_gait_state_origin[l_heel_strike_idx:, :]
            l_gait_state[(frame_num - l_heel_strike_idx):, :] = l_gait_state_origin[0:l_heel_strike_idx, :]
            cross_gait_reward += 0.2 * calc_array_symmetry(r_gait_state_origin, l_gait_state)


        if 'r_cg' in reward_name:
            '''
            2: cross gait
            '''
            reward_str_list.append('r_cg')
            cross_gait_reward += (0.2 / 4.0) * (np.tanh(ankle_to_hip_deg_mat[0, 0]) +
                                                np.tanh(- ankle_to_hip_deg_mat[l_heel_strike_idx, 0]) +
                                                # np.tanh(ankle_to_hip_deg_mat[-1, 0]) + \
                                                np.tanh(-ankle_to_hip_deg_mat[0, 1])
                                                + np.tanh(ankle_to_hip_deg_mat[l_heel_strike_idx, 1])
                                                # + np.tanh(-ankle_to_hip_deg_mat[-1, 1])
                                                )

        # if ankle_to_hip_deg_mat[0, 0] > 5 \
        #         and ankle_to_hip_deg_mat[l_heel_strike_idx, 0] < -5 \
        #         and ankle_to_hip_deg_mat[-1, 0] > 5:
        #     cross_gait_reward += 0.1
        #
        # if ankle_to_hip_deg_mat[0, 1] < -5 \
        #         and ankle_to_hip_deg_mat[l_heel_strike_idx, 1] > 5 \
        #         and ankle_to_hip_deg_mat[-1, 1] < -5:
        #     cross_gait_reward += 0.1
        if 'r_fr' in reward_name:
            '''
            3: foot recovery
            '''
            reward_str_list.append('r_fr')

            ankle_to_hip_speed_mat = np.zeros(ankle_to_hip_deg_mat.shape)
            ankle_to_hip_speed_mat[1:] = ankle_to_hip_deg_mat[1:] - ankle_to_hip_deg_mat[:-1]
            cross_gait_reward += -0.1 * (np.tanh(ankle_to_hip_speed_mat[-1, 0]) +
                                         np.tanh(ankle_to_hip_speed_mat[l_heel_strike_idx, 1]))
        if 'r_po' in reward_name:
            '''
            4: push off
            '''
            reward_str_list.append('r_po')

            r_foot_contact_vec = signal.medfilt(gait_state_mat[:, -2], 3)
            r_foot_contact_vec[1:] -= r_foot_contact_vec[:-1]
            r_foot_contact_vec[0] = 0
            ankle_speed_mat = np.zeros(joint_deg_mat[:, [2, 5]].shape)
            ankle_speed_mat[1:] = joint_deg_mat[1:, [2, 5]] - joint_deg_mat[:-1, [2, 5]]

            if 0 == np.mean(r_foot_contact_vec == -1):
                return cross_gait_reward, reward_str_list
            r_push_off_idx = np.where(r_foot_contact_vec == -1)[0][0]
            cross_gait_reward += -0.1 * np.tanh(ankle_speed_mat[r_push_off_idx, 0])

            if 0 == np.mean(l_foot_contact_vec == -1):
                return cross_gait_reward, reward_str_list
            l_push_off_idx = np.where(l_foot_contact_vec == -1)[0][0]
            cross_gait_reward += -0.1 * np.tanh(ankle_speed_mat[l_push_off_idx, 1])
    return cross_gait_reward, reward_str_list


def calc_gait_symmetry(joint_angle):
    joint_num = int(joint_angle.shape[-1] / 2)
    half_num_sample = int(joint_angle.shape[0] / 2)
    joint_angle_origin = np.copy(joint_angle)
    joint_angle[0:half_num_sample, joint_num:] = joint_angle_origin[half_num_sample:, joint_num:]
    joint_angle[half_num_sample:, joint_num:] = joint_angle_origin[0:half_num_sample, joint_num:]
    dist = np.zeros(joint_num)
    for c in range(joint_num):
        dist[c] = 1 - distance.cosine(joint_angle[:, c], joint_angle[:, c + joint_num])
    return np.mean(dist)

def calc_two_leg_J(joint_state, l_vec):
    '''
    :param q_vec: [q_r_hip, q_r_knee, q_r_ankle, q_l_hip, q_l_knee, q_l_ankle]
    :param l_vec: [l_thigh, l_shank]
    :return: J
    '''
    q_vec_normalized = joint_state[0::2]
    q_vec_denormalized = denormalize_angle(q_vec_normalized)
    J = np.eye(6)
    J[0:2, 0:2] = calc_leg_J(q_vec_denormalized[0:2], l_vec)
    J[3:5, 3:5] = calc_leg_J(q_vec_denormalized[3:5], l_vec)
    return J

def calc_leg_J(q_vec, l_vec, is_polar = False):
    '''
    :param q_vec: [q_hip, q_knee]
    :param l_vec: [l_thigh, l_shank]
    :return: J
    '''
    if is_polar:
        J = np.eye(2)
    else:
        dx_dq_hip  = l_vec[0] * np.cos(q_vec[0]) + l_vec[1] * np.cos(q_vec[0] + q_vec[1])
        dx_dq_knee = l_vec[1] * np.cos(q_vec[0] + q_vec[1])
        dz_dq_hip  = (l_vec[0] * np.sin(q_vec[0]) + l_vec[1] * np.sin(q_vec[0] + q_vec[1]))
        dz_dq_knee = (l_vec[1] * np.sin(q_vec[0] + q_vec[1]))

        J = np.asarray([[dx_dq_hip, dx_dq_knee],
                        [dz_dq_hip, dz_dq_knee]])
    return J


def calc_J_redundent(J):
    '''
    :ref: A comparison of action spaces for learning manipulation tasks, https://arxiv.org/abs/1908.08659
    :param J: end-effector Jacobian
    :return: the force Jacobian from end-point force to joint torques in redundent robot,
    where the freedom of joints is larger than that of end-points
    check the jacobian for the torque.
    '''
    # J^T (JJ^T + alpha I)^-1, avoid singularity
    JT = np.transpose(J)
    JJT = np.matmul(J, JT)
    I = np.eye(J.shape[0])
    JJT_inv = np.linalg.pinv(JJT + 1e-6 * I)
    return np.matmul(JT,JJT_inv)


def joing_angle_2_pos(q_vec, l_vec, is_polar = False):
    '''
    :param q_vec: [q_hip, q_knee, q_ankle]
    :param l_vec: [l_thigh, l_shank]
    :return: pose of ankle: x, z, and q_ankle
    '''
    x_ankle = l_vec[0] * np.sin(q_vec[0]) + l_vec[1] * np.sin(q_vec[0] + q_vec[1])
    z_ankle = -(l_vec[0] * np.cos(q_vec[0]) + l_vec[1] * np.cos(q_vec[0] + q_vec[1]))
    q_ankle = q_vec[2]
    return np.asarray([x_ankle, z_ankle, q_ankle])


def joint_vel_2_end_vel(q_v_vec, q_vec, l_vec, is_polar=False):
    '''
    :param q_v_vec: [q_v_hip, q_v_knee, q_v_ankle]
    :param q_vec: [q_hip, q_knee, q_ankle]
    :param l_vec: [l_thigh, l_shank]
    :return: velocity of ankle: dx, dz, and d_q_ankle
    '''
    J = calc_leg_J(q_vec[0:2], l_vec, is_polar=is_polar)
    vel_ankle = np.zeros(3)
    vel_ankle[0:2] = np.matmul(J, q_v_vec[0:2])
    vel_ankle[-1] = q_v_vec[2]
    return vel_ankle


def state_2_end_state(state, is_polar = False):
    '''

    :param state: 22: [z-z0, cos(error_yaw), sin(error_yaw), v_x, v_y, v_z,
    roll, pitch, q_r_hip, dq_r_hip, q_r_knee, dq_r_knee, q_r_ankle, dq_r_ankle,
    q_l_hip, dq_l_hip, q_l_knee, dq_l_knee, q_l_ankle, dq_l_ankle, foot pressures]
    :return: state_end: this the states of end points: feet and root, including:
    18: [z_m, q_m, v_x, v_z,
    x_r_ankle, z_r_ankle, q_r_ankle, x_l_ankle, z_l_ankle, q_l_ankle,
    dx_r_ankle, dz_r_ankle, dq_r_ankle, dx_l_ankle, dz_l_ankle, dq_l_ankle,
    foot_pressures]
    '''
    joint_states = state[8:-2]
    q_vec_normalized = joint_states[0::2] # normalized to [-1, 1]: 2 * (q - q_mid)/(q_max - q_min)
    q_vec_denormalized = denormalize_angle(q_vec_normalized)
    q_vec = np.copy(q_vec_denormalized)
    q_vec[[2, 5]] = q_vec_normalized[[2, 5]] # q_ankle is not required to calculate Jacobian matrix.

    q_v_vec = joint_states[1::2] # normalized: 0.1 * q_vel
    l_leg = 0.95
    l_th = 0.45 / l_leg # normalized the leg length
    l_sh = 0.5 / l_leg # normalized the leg length
    q_m = state[7]
    z_m = state[0]/l_leg
    v_x = state[3]/l_leg
    v_z = state[5]/l_leg
    l_vec = np.asarray([l_th, l_sh])

    pos_r_ankle = joing_angle_2_pos(q_vec[0:3], l_vec, is_polar=is_polar)
    pos_l_ankle = joing_angle_2_pos(q_vec[3:6], l_vec, is_polar=is_polar)
    vel_r_ankle = joint_vel_2_end_vel(q_v_vec[0:3], q_vec[0:3], l_vec, is_polar=is_polar)
    vel_l_ankle = joint_vel_2_end_vel(q_v_vec[3:6], q_vec[3:6], l_vec, is_polar=is_polar)

    state_end = np.zeros(18)
    state_end[0:4] = np.asarray([z_m, q_m, v_x, v_z])
    state_end[4:16] = np.r_[pos_r_ankle, pos_l_ankle, vel_r_ankle, vel_l_ankle]
    state_end[-2:] = state[-2:]
    return state_end

def denormalize_angle(q_normalized):
    q = np.copy(q_normalized)
    q[[0, 3]] = np.deg2rad(35) + np.deg2rad(80) * q[[0, 3]]
    q[[1, 4]] = np.deg2rad(-75) + np.deg2rad(75) * q[[1, 4]]
    q[[2, 5]] = np.deg2rad(45) * q[[2, 5]]
    return q

def normalize_angle(q):
    q_normalized = np.copy(q)
    q_normalized[:, [0, 3]] = (q_normalized[:, [0, 3]] - np.deg2rad(35)) / np.deg2rad(80)
    q_normalized[:, [1, 4]] = (q_normalized[:, [1, 4]] - np.deg2rad(-75)) / np.deg2rad(75)
    q_normalized[:, [0, 3]] = q_normalized[:, [0, 3]] / np.deg2rad(45)
    return q_normalized


def end_impedance_control(target_end_pos, state, k = 5.0, b = 0.05, k_q = 0.5):
    '''
    :param target_end_pos: [pos_r_ankle, pos_l_ankle]
    :param end_state: [z_m, q_m, v_x, v_z, pos_r_ankle, pos_l_ankle, vel_r_ankle, vel_l_ankle]
    :param joint_state: [q_r_hip, dq_r_hip, q_r_knee, dq_r_knee, q_r_ankle, dq_r_ankle,
    q_l_hip, dq_l_hip, q_l_knee, dq_l_knee, q_l_ankle, dq_l_ankle]
    :param k: 
    :param b: 
    :return:
    '''
    end_state = state_2_end_state(state)
    joint_state = state[8:-2]
    q_vec = joint_state[0::2]
    ankle_pos = end_state[4:10]
    ankle_vel = end_state[10:16]
    ankle_force = k * (target_end_pos - ankle_pos) - b * ankle_vel + 0.5
    l_vec = np.asarray([0.45/0.95, 0.5/0.95])
    J = calc_two_leg_J(joint_state, l_vec)
    torque_offset = k_q * (0.0 - q_vec)
    torque_offset[[0, 3]] = 0 # set the hip torque to 0
    torque = np.matmul(np.transpose(J), ankle_force) + torque_offset
    return torque

def impedance_control(target_pos, joint_states, k = 5.0, b = 0.05):
    q_vec = joint_states[0::2]
    q_v_vec = joint_states[1::2]
    torque = k * (target_pos - q_vec) - b * q_v_vec
    # print('angle error: ', target_pos - q_vec)
    # print('action: ',  action)
    return torque

def calc_torque_from_impedance(action_im, joint_states, scale = 1.0):
    k_vec = action_im[0::3]
    b_vec = action_im[1::3]
    q_e_vec = action_im[2::3]
    q_vec = joint_states[0::2]
    q_v_vec = joint_states[0::2]
    action = (k_vec * (q_e_vec - q_vec) - b_vec * q_v_vec)/scale
    return action


def check_cross_gait(gait_state_mat):
    gait_num_1 = np.mean((gait_state_mat[:, 0] - gait_state_mat[:, 3]) > 0.1)
    gait_num_2 = np.mean((gait_state_mat[:, 0] - gait_state_mat[:, 3]) < -0.1)
    return (gait_num_1 > 0) and (gait_num_2 > 0)


def connect_str_list(str_list):
    if 0 >= len(str_list):
        return ''
    str_out = str_list[0]
    for i in range(1, len(str_list)):
        str_out = str_out + '_' + str_list[i]
    return str_out


def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    len_mean = mean.shape
    log_z = log_std
    z = len_mean[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p


def fifo_data(data_mat, data):
    data_mat[:-1] = data_mat[1:]
    data_mat[-1] = data
    return data_mat


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def joint_state_to_deg(joint_state_mat):
    joint_deg_mat = np.zeros(joint_state_mat.shape)
    joint_deg_mat[:, [0, 3]] = joint_state_mat[:, [0, 3]] * 80.0 + 35.0
    joint_deg_mat[:, [1, 4]] = (1 - joint_state_mat[:, [1, 4]]) * 75.0
    joint_deg_mat[:, [2, 5]] = joint_state_mat[:, [2, 5]] * 45.0
    return joint_deg_mat


def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def plot_joint_angle(joint_angle_resample, human_joint_angle):
    fig, axs = plt.subplots(human_joint_angle.shape[1])
    for c in range(len(axs)):
        axs[c].plot(joint_angle_resample[:, c])
        axs[c].plot(human_joint_angle[:, c])
    plt.legend(['walker 2d', 'human'])
    plt.show()


def read_table(file_name='../../data/joint_angle.xls', sheet_name='walk_fast'):
    dfs = pd.read_excel(file_name, sheet_name=sheet_name)
    data = dfs.values[1:-1, -6:].astype(np.float)
    return data


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def softmax(x):
    # This function is different from the Eq. 17, but it does not matter because
    # both the nominator and denominator are divided by the same value.
    # Equation 17: pi(o|s) = ext(Q^pi - max(Q^pi))/sum(ext(Q^pi - max(Q^pi))
    x_max = np.max(x, axis=-1, keepdims=True)
    e_x = np.exp(x - x_max)
    e_x_sum = np.sum(e_x, axis=-1, keepdims=True)
    out = e_x / e_x_sum
    return out


def write_table(file_name, data):
    df = pd.DataFrame(data)
    df.to_excel(file_name + '.xls', index=False)



