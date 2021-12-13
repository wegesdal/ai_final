from spinup import ddpg_tf1 as ddpg
import tensorflow as tf
import gym

# print(spinup.__file__)

env_fn = lambda : gym.make('Humanoid-v2')

ac_kwargs = dict(hidden_sizes=(256, 256), activation=tf.nn.relu)

logger_kwargs = dict(output_dir='/home/we/ai-final/humanoid/experiment2', exp_name='DDPG')

# seed=0, steps_per_epoch=4000, epochs=100, replay_size=1000000, gamma=0.99, polyak=0.995, pi_lr=0.001, q_lr=0.001, batch_size=100, start_steps=10000, update_after=1000, update_every=50, act_noise=0.1, num_test_episodes=10, max_ep_len=1000, logger_kwargs={}, save_freq=1

ddpg(env_fn=env_fn, epochs=200, ac_kwargs=ac_kwargs, logger_kwargs=logger_kwargs)