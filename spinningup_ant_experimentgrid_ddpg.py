from spinup.utils.run_utils import ExperimentGrid
from spinup import ddpg_tf1 as ddpg
import tensorflow as tf

# seed=0, steps_per_epoch=4000, epochs=100, replay_size=1000000, gamma=0.99, polyak=0.995, pi_lr=0.001, q_lr=0.001, batch_size=100, start_steps=10000, update_after=1000, update_every=50, act_noise=0.1, num_test_episodes=10, max_ep_len=1000, logger_kwargs={}, save_freq=1
eg = ExperimentGrid(name='ant-ddpg-bench')
eg.add('env_name', 'Ant-v2', '', True)
eg.add('epochs', 20)
eg.add('steps_per_epoch', 4000)
eg.add('ac_kwargs:hidden_sizes', (256, 256))
eg.add('gamma', [0.94, 0.99], 'gam')
eg.add('pi_lr', [0.0001, 0.0003, 0.001], 'plr')
eg.add('q_lr', [0.0001, 0.0003, 0.001], 'qlr')
eg.add('batch_size', [50, 100, 200], 'bs')
eg.add('ac_kwargs:activation', tf.nn.relu)
eg.run(ddpg, data_dir='/home/we/ai-final/antgrid/')