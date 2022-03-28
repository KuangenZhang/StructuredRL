import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import glob
from utils.model import LinearActor, LinearCritic, ActorSigmoid, Actor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

class TD3Polar(object):
	def __init__(self, state_dim, action_dim, max_action, lr = 1e-3):
		self.actor = LinearActor(state_dim, action_dim, max_action).to(device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

		self.critic = LinearCritic(state_dim, action_dim).to(device)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

		self.max_action = max_action
		self.it = 0

	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()

	def eval_critic(self, action):
		state = torch.ones(action.shape[0], 1).to(device)
		action = torch.FloatTensor(action).to(device)
		current_Q1 = self.critic(state, action)
		return current_Q1.cpu().data.numpy().flatten()

	def train_once(self, replay_buffer, batch_size=100, policy_freq=2):
		# Sample replay buffer
		x, y, u, r, d, _ = replay_buffer.sample(batch_size=batch_size)
		state = torch.FloatTensor(x).to(device)
		action = torch.FloatTensor(u).to(device)
		next_state = torch.FloatTensor(y).to(device)
		done = torch.FloatTensor(1 - d).to(device)
		reward = torch.FloatTensor(r).to(device)
		for _ in range(100):
			# Get current Q estimates
			current_Q1 = self.critic(state, action)
			# Compute critic loss
			critic_loss = F.mse_loss(current_Q1, reward)
			# print(critic_loss)
			# Optimize the critic
			# print('loss: ', critic_loss)
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()

		for _ in range(100):
			current_Q1 = self.critic(state, self.actor(state))
			actor_loss = -current_Q1.mean()
			# Optimize the actor
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

	def train(self, replay_buffer, batch_size=100, discount=0.99, tau=0.005,
			  policy_noise=0.2, noise_clip=0.5, policy_freq=2):
		self.it += 1
		# for i in range(10):
		self.train_once(replay_buffer, batch_size, policy_freq)

	def save(self, filename, directory):
		torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
		torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

	def load(self, filename, directory):
		actor_path = glob.glob('%s/%s_actor.pth' % (directory, filename))[0]
		self.actor.load_state_dict(torch.load(actor_path))
		critic_path = glob.glob('%s/%s_critic.pth' % (directory, filename))[0]
		print('actor path: {}, critic path: {}'.format(actor_path, critic_path))
		self.critic.load_state_dict(torch.load(critic_path))
