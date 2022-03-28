import os
print(os.getcwd())
import sys
project_path = '../'
sys.path.insert(0, project_path)
sys.path.insert(0, project_path + 'code')
import gym, envs
import numpy as np
import time
import cv2
from mujoco_py import GlfwContext

def main():
    # contro_paras = 0.5 * np.ones(20)
    contro_paras = np.ones(16)
    # env_name = 'CheetahPolarDiyEnv-v0'
    env_name = 'Cheetah2Polar-v0'
    env = gym.make(env_name)

    obs = env.reset()
    done = False
    ave_reward = 0.0

    t = 1
    save_video = False
    render = True
    fps = 50
    if save_video:
        video_name = '../results/video_all/controller.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        GlfwContext(offscreen=True)
        img = env.render(mode='rgb_array')
        out_video = cv2.VideoWriter(video_name, fourcc, fps, (img.shape[1], img.shape[0]))
    if render:
        env.render('human')
    ave_vel = obs[3] * 3
    while not done:
        if 0 == t % 1:
            if save_video:
                img = env.render(mode='rgb_array')
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                out_video.write(img)
            if render:
                env.render('human')
        obs, reward, done, _ = env.step(contro_paras)
        ave_reward += reward
        ave_vel += obs[3] * 3
        # if 0 == t % (1/(fps*env.time_step)):


        t += 1
        # done = False
    if save_video:
        out_video.release()
    # print('V_x: {}'.format(3 * obs[3]))
    print('ave_reward: {}, ave_vel: {}'.format(ave_reward, ave_vel / t))


def step_control(self, control_paras): # best control parameters
    '''
    :param control_paras:in [-1, 1], 15, [k_r, k_theta, b_r, b_theta, delta, l_span, p_x_o, p_y_o, p_x_0, p_x_0, ..., p_x_11, p_x_11]
    :return:
    '''
    self.k_vec = np.asarray([1000, 200]) * control_paras[0:2] * 0.05
    self.b_vec = np.asarray([100, 20]) * control_paras[2:4] * 0.005

    # self.k_vec = np.asarray([1000, 200]) * 0.01
    # self.b_vec = np.asarray([100, 20]) * 0.01
    self.delta = 0.07 * control_paras[4]
    # self.delta = 0.01
    c_points_max = np.asarray([[-0.2, -0.5], [-0.2805, -0.5], [-0.3, -0.361], [-0.3, -0.361],
                               [-0.3, -0.361], [0.0, -0.361], [0.0, -0.361], [0.0, -0.3214],
                               [0.3032, -0.3214], [0.3032, -0.3214], [0.2826, -0.50], [0.2, -0.5]])  # m
    c_points_paras = control_paras[5:-2]
    # c_points_paras[c_points_paras < 0.2] = 0.2
    c_points_ratio = np.asarray([[c_points_paras[0], c_points_paras[9]], [c_points_paras[1], c_points_paras[9]],
                                 [c_points_paras[2], c_points_paras[7]], [c_points_paras[2], c_points_paras[7]],
                                 [c_points_paras[2], c_points_paras[7]], [c_points_paras[3], c_points_paras[7]],
                                 [c_points_paras[3], c_points_paras[7]], [c_points_paras[3], c_points_paras[8]],
                                 [c_points_paras[4], c_points_paras[8]], [c_points_paras[4], c_points_paras[8]],
                                 [c_points_paras[5], c_points_paras[9]], [c_points_paras[6], c_points_paras[9]]])

    self.c_points = c_points_max * c_points_ratio

    self.p_o = 0.5 * (self.c_points[0] + self.c_points[-1])
    self.l_span = self.p_o[0] - self.c_points[0, 0]

    # self.T_st = 0.25 * (2.0 * control_paras[-2] + 1e-3)  # s
    # self.T_sw = 0.25 * (2.0 * control_paras[-1] + 1e-3)
    self.T_st = (2 * self.l_span / self.v_d) * (1.0 * control_paras[-2]) # s
    self.T_sw = 0.25 * (0.6 * control_paras[-1])
    self.T_stride = self.T_st + self.T_sw
    torque = self.leg_control
    # print(torque)
    return torque

if __name__ == "__main__":
    main()