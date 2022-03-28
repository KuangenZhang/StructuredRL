from gym import utils, error, spaces
from gym.envs.mujoco import mujoco_env
from envs.controller import ImControl
import shutil
import os
import numpy as np

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

class Cheetah2Env(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, robot_file='mit_cheetah_mujoco.xml'):
        self.v_d = 6.0
        self.scale = 1.0
        self.f_x = 0
        self.is_calc_done = True

        env_path = os.path.dirname(mujoco_env.__file__)
        shutil.copyfile('../data/robot_model/mit_cheetah/{}'.format(robot_file),
                 '{}/assets/{}'.format(env_path, robot_file))
        mujoco_env.MujocoEnv.__init__(self, robot_file, 5)
        utils.EzPickle.__init__(self)

    def update_v_d(self, v_d):
        self.v_d =  v_d

    def update_external_force(self, f_x):
        external_force = np.zeros(6)
        external_force[0] = f_x
        self.sim.data.xfrc_applied[:] = external_force

    def update_calc_done(self, is_calc_done):
        self.is_calc_done = is_calc_done

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        v = (xposafter - xposbefore) / self.dt
        reward = self.calc_reward(v, action)
        if self.is_calc_done:
            done = self.calc_done()
        else:
            done = False
        debug = False
        if debug:
            # print('pitch: ', pitch)
            # print('z: ', height)
            # print('action: ', action)
            # print('self.v_d: ', self.v_d)
            print('v: ', v)
        return ob, reward, done, {}

    def get_torque(self):
        return self.sim.data.ctrl

    def calc_done(self):
        height = self.sim.data.qpos[1]
        pitch = self.sim.data.qpos[2]
        done = abs(pitch) > 1.0 or height < -0.1 * self.scale
        return done

    def calc_reward(self, v, action):
        reward_run = v
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward = reward_run + reward_ctrl
        return reward

    def _get_obs(self):
        obs = np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat, self.calc_foot_contact().flat])
        return obs

    def reset_model(self):
        q_vec = self.init_qpos
        q_vel = 0.0 * self.init_qvel
        self.set_state(q_vec, q_vel)
        return self._get_obs()

    def calc_foot_contact(self, foot_list=None):
        if foot_list is None:
            foot_list = ['flfoot', 'blfoot', 'frfoot', 'brfoot']
        foot_contact = np.zeros(len(foot_list))
        for i in range(self.sim.data.ncon):
            # Note that the contact array has more than `ncon` entries,
            # so be careful to only read the valid entries.
            contact = self.sim.data.contact[i]
            geom2_name = self.sim.model.geom_id2name(contact.geom2)
            if geom2_name in foot_list:
                foot_contact[foot_list.index(geom2_name)] = 1

            debug = False
            if debug:
                print('id', i)
                print('geom1', contact.geom1, self.sim.model.geom_id2name(contact.geom1))
                print('geom2', contact.geom2, self.sim.model.geom_id2name(contact.geom2))
                c_array = np.zeros(6, dtype=np.float64)
                mujoco_py.functions.mj_contactForce(self.sim.model, self.sim.data, i, c_array)
                print('contact force', c_array)

        return foot_contact


class Cheetah2PolarEnv(Cheetah2Env, ImControl):
    def __init__(self, robot_file='mit_cheetah_mujoco.xml', paras_num = 16):
        self.gait_event = np.zeros(4)
        self.control_time = 0.0
        self.paras_num = paras_num
        self.time_step = 0
        self.max_time_steps = 1000
        self.scale = 1.0
        self.v_d = 6.0
        Cheetah2Env.__init__(self, robot_file=robot_file)

    def reset_model(self):
        self.gait_event = np.zeros(4)
        self.control_time = 0.0
        self.time_step = 0
        return Cheetah2Env.reset_model(self)

    def step(self, control_paras):
        # convert control parameters to torques
        action = self.step_control(control_paras)
        gait_event_before = self.calc_foot_contact()
        ob, reward, done, _ = Cheetah2Env.step(self, action)
        gait_event_after = self.calc_foot_contact()
        self.gait_event = gait_event_after - gait_event_before
        self.update_control_time()
        debug = False
        if debug:
            print('action: ', action)
            print('gait event: ', self.gait_event)
        self.time_step += 1
        return ob, reward, done, {}

    def _set_action_space(self):
        self.action_space = spaces.Box(low=np.zeros(self.paras_num), high=np.ones(self.paras_num), dtype=np.float32)
        return self.action_space

    def update_control_time(self):
        gait_event = self.gait_event
        # collision of front left foot
        if 1 == gait_event[0] and self.control_time > 0.9 * self.T_stride:
            self.control_time = 0.0
            # self.v_d = 0.5 + 5 * self.time_step / self.max_time_steps
        else:
            self.control_time += self.dt
        if self.control_time >= self.T_stride:
            self.control_time = self.T_stride

    def calc_origin_angle(self, q_vec_offset):
        q_vec = np.copy(q_vec_offset)
        q_vec[0::3] += np.deg2rad(20)
        q_vec[1::3] += np.deg2rad(-90)
        q_vec[2::3] += np.deg2rad(115)
        return q_vec

    def leg_control(self, l_mat = None, front_eq_back = False):
        q_vec_all = self.calc_origin_angle(self.sim.data.qpos[3:])
        q_v_vec_all = self.sim.data.qvel[3:]
        delta_t = np.asarray([0, 0.5 * self.T_stride,
                              0.5 * self.T_stride, 0])
        if l_mat is None:
            # fl, bl, fr, br
            l_mat = self.scale * np.asarray([[0.162, 0.215, 0.221],
                                 [0.244, 0.215, 0.167]])
        leg_num = 4
        joint_num = l_mat.shape[-1]
        torque_vec = np.zeros(joint_num * leg_num)
        for r in range(2):  # l, r
            for c in range(2):  # f, b
                i = 2 * r + c
                indices = np.arange(joint_num * i, joint_num * (i + 1))
                q_vec = q_vec_all[indices]
                q_v_vec = q_v_vec_all[indices]
                if front_eq_back:
                    target_pos, target_vel = self.calc_target_point(self.control_time - delta_t[i], 0)
                else:
                    target_pos, target_vel = self.calc_target_point(self.control_time - delta_t[i], c)
                torque_vec[indices] = self.polar_impedance_control(
                    q_vec, q_v_vec, target_pos, target_vel,
                    self.k_vec, self.b_vec, l_vec=l_mat[c])
        torque_vec = np.clip(torque_vec, -1, 1)
        return torque_vec


class MiniCheetahPolarEnv(Cheetah2PolarEnv):
    def __init__(self, robot_file = 'mit_mini_cheetah_mujoco.xml', paras_num = 16):
        Cheetah2PolarEnv.__init__(self, robot_file=robot_file, paras_num = paras_num)

    def leg_control(self, l_mat=None):
        if l_mat is None:
            # fl, bl, fr, br
            l_mat = self.scale * 0.283 * np.ones((2, 2))
        return Cheetah2PolarEnv.leg_control(self, l_mat, front_eq_back = True)

    def calc_origin_angle(self, q_vec_offset):
        q_vec = np.copy(q_vec_offset)
        q_vec[0::2] += np.deg2rad(-45)
        q_vec[1::2] += np.deg2rad(90)
        return q_vec

    def calc_foot_contact(self, foot_list=None):
        if foot_list is None:
            foot_list = ['flshin', 'blshin', 'frshin', 'brshin']

        return Cheetah2PolarEnv.calc_foot_contact(self, foot_list)

    def step_control(self, control_paras):
        '''
        :param control_paras:in [-1, 1], 15, [k_r, k_theta, b_r, b_theta, delta, l_span, p_x_o, p_y_o, p_x_0, p_x_0, ..., p_x_11, p_x_11]
        :return:
        '''
        control_paras = np.copy(control_paras)
        self.k_vec = np.asarray([50, 1]) * (3.0 * control_paras[0:2] + 1e-2)
        self.b_vec = np.asarray([2, 0.08]) * (3.0 * control_paras[2:4] + 1e-2)

        self.delta = self.scale * 0.04 * control_paras[4]
        c_points_max = np.asarray([[-0.2  , -0.5  ],
                                   [-0.3  , -0.5  ],
                                   [-0.35 , -0.35 ],
                                   [-0.35 , -0.35 ],
                                   [-0.35 , -0.35 ],
                                   [0.0   , -0.35 ],
                                   [0.0   , -0.35 ],
                                   [0.0   , -0.30 ],
                                   [0.35  , -0.30 ],
                                   [0.35  , -0.30 ],
                                   [0.3   , -0.5  ],
                                   [0.2   , -0.5  ]])  # m
        c_points_paras = 0.5 * control_paras[5:-1] + 0.6
        # c_points_paras[c_points_paras < 0.2] = 0.2
        c_points_ratio = np.asarray([[c_points_paras[0], c_points_paras[9]], [c_points_paras[1], c_points_paras[9]],
                                     [c_points_paras[2], c_points_paras[7]], [c_points_paras[2], c_points_paras[7]],
                                     [c_points_paras[2], c_points_paras[7]], [c_points_paras[3], c_points_paras[7]],
                                     [c_points_paras[3], c_points_paras[7]], [c_points_paras[3], c_points_paras[8]],
                                     [c_points_paras[4], c_points_paras[8]], [c_points_paras[4], c_points_paras[8]],
                                     [c_points_paras[5], c_points_paras[9]], [c_points_paras[6], c_points_paras[9]]])

        self.c_points = self.scale * c_points_max * c_points_ratio

        self.p_o = 0.5 * (self.c_points[0] + self.c_points[-1])
        self.l_span = self.p_o[0] - self.c_points[0, 0]

        # self.T_st = (2 * self.l_span / self.v_d) * (1.0 * control_paras[-2] + 0.1) # s
        # self.T_sw = 0.25 * (1.0 * control_paras[-1] + 0.1)
        self.T_st = (2 * self.l_span / self.v_d) * (1.0 * control_paras[-1] + 0.1) # s
        self.T_sw = 0.25 * (1.0 * control_paras[-1] + 0.1)
        self.T_stride = self.T_st + self.T_sw
        torque = self.leg_control()
        # if self.time_step < 1:
        #     print('k: {} \n'.format(self.k_vec) +
        #           'b: {} \n'.format(self.b_vec) +
        #           'delta :{}\n'.format(self.delta) +
        #           'Tst: {}\n'.format(self.T_st) +
        #           'Tsw: {}\n'.format(self.T_sw) +
        #           'vd: {}\n'.format(self.v_d) +
        #           'control_paras: {}\n'.format(control_paras) +
        #           'c_points: {}\n'.format(self.c_points)
        #           )
        return torque


class MiniCheetahPolarVdEnv(MiniCheetahPolarEnv):
    def __init__(self, robot_file = 'mit_mini_cheetah_mujoco.xml', paras_num = 16):
        MiniCheetahPolarEnv.__init__(self, robot_file=robot_file, paras_num = paras_num)

    def calc_reward(self, v, action, ctrl_ratio = 0.01):
        reward_run = 1 - np.abs(self.v_d - v)/self.v_d
        joint_q_vel = self.sim.data.qvel.flat[3:]
        reward_ctrl = - ctrl_ratio * np.abs(action * joint_q_vel).sum()
        # reward_ctrl = - ctrl_ratio * np.square(action).sum()
        reward = reward_run + reward_ctrl
        return reward

    def reset_model(self):
        self.v_d = np.random.uniform(low=1.0, high=5.0)
        return MiniCheetahPolarEnv.reset_model(self)

    def _get_obs(self):
        obs = np.concatenate([np.asarray([self.v_d]),
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat, self.calc_foot_contact().flat])
        return obs


