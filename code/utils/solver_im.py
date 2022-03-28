import numpy as np
import os
import datetime
import cv2
import torch
import glob
import time
import shutil
import gym
from utils import utils
from tqdm import tqdm
from scipy.stats import multivariate_normal
from tensorboardX import SummaryWriter
from shutil import copyfile
from scipy import signal
from methods import TD3Polar

from mujoco_py import GlfwContext

# train the actor to fit the data
class Solver(object):
    def __init__(self, args, env, project_path):
        print(args)
        self.args = args
        self.env = env
        self.file_name = ''
        self.project_path = project_path
        self.result_path = project_path + "results"

        self.evaluations = []
        self.estimate_Q_vals = []
        self.Q1_vec = []
        self.Q2_vec = []
        self.true_Q_vals = []
        self.Q_ae_mean_vec = []
        self.Q_ae_std_vec = []


        self.env.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

        # Initialize policy
        if 'TD3Polar' == args.policy_name:
            policy = TD3Polar.TD3Polar(1, action_dim, max_action, lr = args.lr)
        elif 'TD3Sigmoid' == args.policy_name:
            policy = TD3Sigmoid.TD3Sigmoid(state_dim, action_dim, max_action, lr=args.lr)
        else:
            policy = TD3.TD3(state_dim, action_dim, max_action, lr = args.lr)

        self.policy = policy
        self.policy_name = self.policy.__class__.__name__
        print('-------Current policy: {} --------------'.format(self.policy_name))
        # self.replay_buffer = utils.ReplayBuffer(max_size=1e4)
        self.replay_buffer = utils.ReplayBufferMat(max_size=args.max_time_steps)
        self.total_time_steps = 0
        self.episode_timesteps = 0
        self.pre_num_steps = self.total_time_steps
        self.time_steps_since_eval = 0
        self.time_steps_calc_Q_vale = 0
        self.best_reward = -1e4

        self.env_timeStep = 5

    def train_once(self):
        if self.total_time_steps > self.args.eval_freq:
            self.policy.train(self.replay_buffer, self.args.batch_size, self.args.discount,
                              self.args.tau, self.args.policy_noise, self.args.noise_clip,
                              self.args.policy_freq)

    def eval_once(self):
        self.pbar.update(self.total_time_steps - self.pre_num_steps)
        self.pre_num_steps = self.total_time_steps

        # Evaluate episode
        if self.time_steps_since_eval >= self.args.eval_freq:
            self.time_steps_since_eval %= self.args.eval_freq
            avg_reward = evaluate_policy(self.env, self.policy, self.args)
            self.evaluations.append(avg_reward)
            self.writer_test.add_scalar('ave_reward', avg_reward, self.total_time_steps)

            if self.args.save_all_policy:
                self.policy.save(
                    self.file_name + str(int(int(self.total_time_steps / self.args.eval_freq) * self.args.eval_freq)),
                    directory=self.log_dir)

            np.save(self.log_dir + "/test_accuracy", self.evaluations)
            utils.write_table(self.log_dir + "/test_accuracy", np.asarray(self.evaluations))

            if self.best_reward < avg_reward:
                self.best_reward = avg_reward
                # print("-------------------Best reward!----------------------")
                self.policy.save(self.file_name, directory=self.log_dir)
                print("Best reward! Total T: %d Reward: %f" %
                      (self.total_time_steps, avg_reward))

            # self.policy.save(self.file_name, directory=self.log_dir)
            # print("Total T: %d Reward: %f" %
            #       (self.total_time_steps, avg_reward))

    def reset(self):
        # Reset environment
        self.obs = self.env.reset()
        self.obs_vec = np.dot(np.ones((self.args.seq_len, 1)), self.obs.reshape((1, -1)))
        self.episode_timesteps = 0
        self.still_steps = 0

    def step_episode(self, action):
        # Perform action
        ave_reward = 0
        done = False
        while not done:
            new_obs, reward, done, _ = self.env.step(action)
            ave_reward += reward
            self.episode_timesteps += 1
            self.total_time_steps += 1
            self.time_steps_since_eval += 1
            self.time_steps_calc_Q_vale += 1
        return new_obs, ave_reward, done, {}

    def step_k_steps(self, action, k = 1):
        ave_reward = 0
        finish = False
        # while not finish:
        for _ in range(k):
            new_obs, reward, done, _ = self.env.step(action)
            ave_reward += reward
            self.episode_timesteps += 1
            self.total_time_steps += 1
            self.time_steps_since_eval += 1
            self.time_steps_calc_Q_vale += 1
            # finish = done or 0 == self.env.control_time
            if done:
                break
        # print('0 == new_obs[-1]: ', 0 == new_obs[-1])
        return new_obs, ave_reward, done, {}


    def pre_train(self):
        data = np.load('../data/im_motion/walk_dataset.npy', allow_pickle=True).item()
        states = data['state']
        actions = data['action']
        train_loader, valid_loader, test_loader, states_test, actions_test = \
            utils.generate_dataset(states, actions, self.args.batch_size, shuffle=True)
        print('Length of training set: {}, validation set: {}, and test set: {}'.format(
            len(train_loader.dataset), len(valid_loader.dataset), len(test_loader.dataset)))

        pre_loss = 1.0
        for t in range(100):
            train_loss, valid_loss = self.policy.train_expert(train_loader, valid_loader)
            if valid_loss < pre_loss:
                print('Training loss: {}, valid loss: {}'.format(train_loss, valid_loss))
                pre_loss = valid_loss
                self.policy.save(self.file_name, directory=self.log_dir)

        self.policy.load(self.file_name, directory=self.log_dir)
        self.policy.transfer_expert()
        print('Finish pre-training.')

    def save_sample(self, sample_num = int(1e3)):
        self.log_dir = '{}/{}/reward_samples'.format(self.result_path, self.args.log_path)
        print("---------------------------------------")
        print("Settings: %s" % self.log_dir)
        print("---------------------------------------")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        action_num = self.env.action_space.shape[0]
        self.pbar = tqdm(total=sample_num * action_num, initial=0, position=0, leave=True)
        sample_buffer = np.zeros((sample_num * action_num,
                                  action_num + 1))
        for r in range(sample_num):
            for c in range(action_num):
                self.pbar.update(1)
                self.env.reset()
                action = 0.5 * np.ones(action_num)
                action[c] = np.random.uniform(0, 1)
                # action = self.env.action_space.sample()
                new_obs, reward, done, _ = self.step_episode(action)
                idx = r * action_num + c
                sample_buffer[idx, :-1] = action
                sample_buffer[idx, -1] = reward
                if 0 == idx % 20:
                    np.save(self.log_dir + "/reward_samples.npy", sample_buffer)

    def train(self):
        # Evaluate untrained policy
        self.evaluations = [evaluate_policy(self.env, self.policy, self.args)]
        self.log_dir = '{}/{}/{}_{}_seed_{}'.format(self.result_path, self.args.log_path,
                                                    self.args.policy_name, self.args.env_name,
                                                    self.args.seed)
        print("---------------------------------------")
        print("Settings: %s" % self.log_dir)
        print("---------------------------------------")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        # copy solver, TD3, and DAC to the log_dir
        copyfile('methods/{}.py'.format(self.policy_name), '{}/{}.py'.format(self.log_dir, self.policy_name))
        copyfile('utils/solver_im.py', '{}/solver_im.py'.format(self.log_dir))
        copyfile('utils/utils.py', '{}/utils.py'.format(self.log_dir))
        copyfile('envs/cheetah_mujoco.py', '{}/cheetah_mujoco.py'.format(self.log_dir))
        copyfile('main_im.py', '{}/main_im.py'.format(self.log_dir))

        # TesnorboardX
        self.writer_test = SummaryWriter(logdir=self.log_dir)
        self.pbar = tqdm(total=self.args.max_time_steps, initial=self.total_time_steps, position=0, leave=True)
        done = True

        if 'IM' in self.args.policy_name and 'End' not in self.args.policy_name:
            self.pre_train() # train_expert

        while self.total_time_steps < self.args.max_time_steps:
            # ================ Train =============================================#
            self.train_once()
            # ====================================================================#
            if done:
                self.eval_once()
                self.reset()
            # Select action randomly or according to policy
            if self.total_time_steps < self.args.start_time_steps:
                # print(self.env.action_space.low, self.env.action_space.high)
                action = self.env.action_space.sample()
            else:
                if 'Polar' in self.args.policy_name:
                    action = self.policy.select_action(np.asarray([self.obs[0]]))
                else:
                    action = self.policy.select_action(np.array(self.obs))

                # if np.random.uniform(0, 1) < self.args.expl_noise:
                #     action = self.env.action_space.sample()
                # else:
                noise = np.random.normal(0, self.args.expl_noise,
                                         size=action.shape[0])
                action = (action + noise)

            if 'Polar' in self.args.policy_name or 'k_steps' in self.args.policy_name:
                action = action.clip(0, 1)
            else:
                action = action.clip(-1, 1)

            state_id = 0

            # Perform action
            if 'Polar' in self.args.policy_name:
                new_obs, reward, done, _ = self.step_episode(action)
            # elif 'k_steps' in self.args.policy_name:
            #     new_obs, reward, done, _ = self.step_k_steps(action)
            else:
                new_obs, reward, done, _ = self.env.step(action)
                self.episode_timesteps += 1
                self.total_time_steps += 1
                self.time_steps_since_eval += 1
                self.time_steps_calc_Q_vale += 1


            done_bool = 0 if self.episode_timesteps >= self.env._max_episode_steps else float(done)

            if 'Polar' in self.args.policy_name:
                self.replay_buffer.add((np.asarray([self.obs[0]]),
                                        np.asarray([new_obs[0]]), action, reward,
                                        done_bool, state_id))
            else:
                self.replay_buffer.add((self.obs,
                                        new_obs, action, reward,
                                        done_bool, state_id))

            self.obs = new_obs

        # Final evaluation
        self.eval_once()
        self.env.reset()

    def eval_only(self, is_reset = True):
        video_dir = '{}/video_all/video_IM/{}_{}'.format(
            self.result_path, self.args.policy_name, self.args.env_name)
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        model_path_vec = glob.glob(self.result_path + '/{}/{}_{}_seed*'.format(
            self.args.log_path, self.args.policy_name, self.args.env_name))
        print(model_path_vec)
        fps = 50
        self.env.render(mode='human')
        for model_path in model_path_vec:
            # print(model_path)
            self.policy.load("%s" % (self.file_name + self.args.load_policy_idx), directory=model_path)

            video_name = video_dir + '/{}_{}_{}.mp4'.format(
                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                self.file_name, self.args.load_policy_idx)
            obs = self.env.reset()

            obs_mat = np.asarray(obs)
            action_mat = None


            time.sleep(1)
            use_mujoco = True
            # if use_mujoco and self.args.save_video:
            #     GlfwContext(offscreen=True)
            done = False
            ave_reward = 0.0
            ave_vel = obs[3] * 3
            if self.args.save_video:
                fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                img = self.env.render(mode='rgb_array')
                out_video = cv2.VideoWriter(video_name, fourcc,
                                            fps, (img.shape[1], img.shape[0]))

            t = 0
            while not done:
                obs = self.add_disturbance(t, obs)
                if 'Polar' in self.args.policy_name:
                    action = self.policy.select_action(np.asarray([obs[0]]))
                else:
                    action = self.policy.select_action(np.array(obs))

                obs, reward, done, _ = self.env.step(action)
                ave_reward += reward
                obs_mat = np.c_[obs_mat, np.asarray(obs)]
                if action_mat is None:
                    action_mat = np.asarray(self.env.get_torque())
                else:
                    action_mat = np.c_[action_mat, np.asarray(self.env.get_torque())]
                if self.args.save_video:
                    if 0 == t % (1 / (fps * self.env.dt)):
                        img = self.env.render(mode='rgb_array')
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        out_video.write(img)
                elif self.args.render:
                    self.env.render(mode='human')
                t += 1

            if not self.args.render:
                utils.write_table(video_name + '_state', np.transpose(obs_mat))
                utils.write_table(video_name + '_torque', np.transpose(action_mat))
            if self.args.save_video:
                out_video.release()
            print('Average reward: {}'.format(ave_reward))
        if is_reset:
            self.env.reset()

    def add_disturbance(self, t, obs):
        if t < 400:
            v_d = 2
        elif t < 800:
            v_d = 4
        else:
            v_d = 3
        self.env.update_v_d(v_d)
        obs[0] = v_d
        if t > 1200 and t < 1300:
            self.env.update_external_force(f_x = self.env.scale * 10)
        elif t > 1600 and t < 1700:
            self.env.update_external_force(f_x = -self.env.scale * 10)
        else:
            self.env.update_external_force(f_x=0)
        return obs


# Runs policy for X episodes and returns average reward
def evaluate_policy(env, policy, args, eval_episodes=10):
    avg_reward = 0.
    for i in range(eval_episodes):
        obs = env.reset()
        done = False
        t = 0
        if 'Polar' in args.policy_name:
            action = policy.select_action(np.asarray([obs[0]]))
        while not done:
            if 'Polar' not in args.policy_name:
                # if 'k_steps' in args.policy_name:
                #     # if 0 == t % 100:
                #     if 0 == env.control_time:
                #         action = policy.select_action(np.array(obs))
                # else:
                action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
            t += 1
    avg_reward /= eval_episodes
    return avg_reward