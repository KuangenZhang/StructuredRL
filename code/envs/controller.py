from scipy.special import comb
import numpy as np

class ImControl():
    def __init__(self):
        self.gait_event = np.zeros(4)
        self.control_time = 0.0
        self.v_d = 6.0
        self.paras_num = 16
        self.dt = 0
        self.max_time_steps = 5000
        self.scale = 1.0

    def step_control(self, control_paras):
        '''
        :param control_paras:in [-1, 1], 15, [k_r, k_theta, b_r, b_theta, delta, l_span, p_x_o, p_y_o, p_x_0, p_x_0, ..., p_x_11, p_x_11]
        :return:
        '''
        self.k_vec = np.asarray([50, 1]) * (3.0 * control_paras[0:2] + 0.1)
        self.b_vec = np.asarray([1, 0.04]) * (3.0 * control_paras[2:4] + 0.1)

        self.delta = self.scale * 0.08 * control_paras[4]
        c_points_max = np.asarray([[-0.2, -0.5], [-0.3, -0.5], [-0.35, -0.35], [-0.35, -0.35],
                                   [-0.35, -0.35], [0.0, -0.35], [0.0, -0.35], [0.0, -0.30],
                                   [0.35, -0.30], [0.35, -0.30], [0.3, -0.5], [0.2, -0.5]])  # m
        c_points_paras = 0.4 * control_paras[5:-1] + 0.8
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
        return torque

    def leg_control(self, leg_num=4, q_vec_all=None, q_v_vec_all=None,
                    l_mat=None, delta_t=None):
        """
        Control the torques of each joint.
        Implement this in each subclass.
        """
        raise NotImplementedError

    def calc_target_point(self, t, leg_idx):
        if 0 <= t and t <= self.T_st:
            s_st = t / self.T_st
            delta = self.delta
            if 1 == leg_idx:
                delta = 0
            target_pos, target_vel = self.sin_curve(s_st, self.l_span, delta, self.p_o, self.T_st)
        else:
            if t < 0:
                s_sw = (t + self.T_sw) / self.T_sw
            else:
                s_sw = (t - self.T_st) / self.T_sw
            target_pos, target_vel = self.bezier_curve(s_sw, self.c_points, self.T_sw)
        # t = np.mod(t, self.T_stride)
        # if t <= self.T_st:
        #     s_st = t / self.T_st
        #     delta = self.delta
        #     if 1 == leg_idx:
        #         delta = 0
        #     target_pos, target_vel = self.sin_curve(s_st, self.l_span, delta, self.p_o, self.T_st)
        # else:
        #     s_sw = (t - self.T_st) / self.T_sw
        #     target_pos, target_vel = self.bezier_curve(s_sw, self.c_points, self.T_sw)
        return target_pos, target_vel

    def bernstein_poly(self, k, n, s):
        """
         The Bernstein polynomial of n, k as a function of t
        """

        return comb(n, k) * (s ** k) * (1 - s) ** (n - k)

    def bezier_curve(self, s, points, T_sw):
        """
            Given a set of control points, return the
            bezier curve defined by the control points.

            points should be a 2d numpy array:
                   [ [1,1],
                     [2,3],
                     [4,5], ..[Xn, Yn] ]
            s in [0, 1] is the current phase
            See https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/Bezier/bezier-der.html
            :return
        """
        n = points.shape[0] - 1
        B_vec = np.zeros((1, n + 1))
        for k in range(n + 1):
            B_vec[0, k] = self.bernstein_poly(k, n, s)
        target_pos = np.matmul(B_vec, points).squeeze()
        d_points = points[1:] - points[:-1]
        target_vel = (1/T_sw) * n * np.matmul(B_vec[:, :-1], d_points).squeeze()
        return target_pos, target_vel

    def sin_curve(self, s, l_span, delta, p_st_0, T_st):
        '''
        :param l_span: half of the stroke length
        :param delta: penetrating amplitude
        :param s: in [0, 1], the current phase
        :return: the target stance point at the current phase s
        '''
        p_x = l_span * (1 - 2 * s) + p_st_0[0]
        target_pos = np.asarray([p_x, -delta * np.cos(np.pi * (p_x - p_st_0[0]) / (2 * l_span)) + p_st_0[1]])
        target_vel = np.asarray([-2 * l_span / T_st,
                                 (-delta * np.pi / T_st) * np.sin(np.pi * (p_x - p_st_0[0]) / (2 * l_span))])
        return target_pos, target_vel

    def joint_2_cartesian_position(self, q_vec, l_vec):
        '''
        :param q_vec: [q_hip, q_knee, q_ankle]
        :param l_vec: [l_thigh, l_shank, l_foot]
        :return: [x z]
        '''
        x = 0
        z = 0
        for i in range(len(q_vec)):
            q_sum = np.sum(q_vec[:i+1])
            x += l_vec[i] * np.sin(q_sum)
            z += -(l_vec[i] * np.cos(q_sum))
        return np.asarray([x, z])

    def joint_2_cartesian_Jacobian(self, q_vec, l_vec):
        '''
        :param q_vec: [q_hip, q_knee, q_ankle]
        :param l_vec: [l_thigh, l_shank, l_foot]
        :return: J
        '''
        J = np.zeros((2, len(q_vec)))
        for r in range(len(q_vec)):
            for c in range(r, len(q_vec)):
                q_sum = np.sum(q_vec[:c+1])
                J[0, r] += l_vec[c] * np.cos(q_sum)
                J[1, r] += l_vec[c] * np.sin(q_sum)
        return J


    def cartesian_2_polar_position(self, p_vec):
        '''
        :param p_vec: [x z]
        :return: [r, beta]
        '''
        x, z = p_vec
        r = np.sqrt(x ** 2 + z ** 2)
        beta = np.arctan2(x, -z)
        return np.asarray([r, beta])

    def cartesian_2_polar_Jacobian(self, p_vec):
        '''
        :param p_vec: [x z]
        :return: J
        '''
        x, z = p_vec
        J = np.eye(2)
        if 0 == x and z == 0:
            return J
        dr_dx = x / (np.sqrt(x ** 2 + z ** 2))
        dr_dz = z / (np.sqrt(x ** 2 + z ** 2))
        dbeta_dx = -z / (x ** 2 + z ** 2)
        dbeta_dz = x / (x ** 2 + z ** 2)
        J = np.asarray([[dr_dx, dr_dz],
                        [dbeta_dx, dbeta_dz]])
        return J

    def polar_impedance_control(self, q_vec, q_v_vec, target_pos, target_vel, k_vec, b_vec,
                                l_vec=np.asarray([0.35, 0.35])):
        p_vec = self.joint_2_cartesian_position(q_vec, l_vec)
        polar_vec = self.cartesian_2_polar_position(p_vec)
        e_pos_vec = polar_vec - self.cartesian_2_polar_position(target_pos)
        J_joint_2_polar = np.matmul(self.cartesian_2_polar_Jacobian(p_vec),
                                    self.joint_2_cartesian_Jacobian(q_vec, l_vec))
        polar_vel_vec = np.matmul(J_joint_2_polar, q_v_vec)
        e_vel_vec = polar_vel_vec - np.matmul(self.cartesian_2_polar_Jacobian(target_pos),
                                              target_vel)
        force_polar = -(k_vec * e_pos_vec + b_vec * e_vel_vec)
        torque = np.matmul(np.transpose(J_joint_2_polar), force_polar)
        return torque