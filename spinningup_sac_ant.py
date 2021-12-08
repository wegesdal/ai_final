from spinup import sac_tf1 as sac
import tensorflow as tf
import gym

# print(spinup.__file__)

env_fn = lambda : gym.make('Ant-v2')

ac_kwargs = dict(hidden_sizes=(256, 256), activation=tf.nn.relu)

logger_kwargs = dict(output_dir='/home/we/ai-final/logs', exp_name='final_proj_test')

# dict(steps_per_epoch=4000, epochs=100, replay_size=1000000, gamma=0.99, polyak=0.995, lr=0.001, alpha=0.2, batch_size=100, start_steps=10000, update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000)

sac(env_fn=env_fn, epochs=1, ac_kwargs=ac_kwargs, logger_kwargs=logger_kwargs)