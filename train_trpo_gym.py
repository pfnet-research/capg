from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

import argparse
import logging
import os

import chainer
from chainer import functions as F
import gym
gym.undo_logger_setup()
import gym.wrappers
import numpy as np

import chainerrl

from clipped_gaussian import ClippedGaussian

from call_render import CallRender
from clip_action import ClipAction


class ClippedGaussianPolicy(
        chainerrl.policies.FCGaussianPolicyWithStateIndependentCovariance):

    def __call__(self, x):
        mean = self.hidden_layers(x)
        var = F.broadcast_to(self.var_func(self.var_param), mean.shape)
        assert self.min_action is not None
        assert self.max_action is not None
        return ClippedGaussian(mean, var,
                               low=self.xp.asarray(self.min_action),
                               high=self.xp.asarray(self.max_action))


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1,
                        help='GPU device ID. Set to -1 to use CPUs only.')
    parser.add_argument('--env', type=str, default='Hopper-v1',
                        help='Gym Env ID')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 32)')
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
    parser.add_argument('--steps', type=int, default=10 ** 6,
                        help='Total time steps for training.')
    parser.add_argument('--eval-interval', type=int, default=100000,
                        help='Interval between evaluation phases in steps.')
    parser.add_argument('--eval-n-runs', type=int, default=100,
                        help='Number of episodes ran in an evaluation phase')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Render the env')
    parser.add_argument('--demo', action='store_true', default=False,
                        help='Run demo episodes, not training')
    parser.add_argument('--load', type=str, default='',
                        help='Directory path to load a saved agent data from'
                             ' if it is a non-empty string.')
    parser.add_argument('--trpo-update-interval', type=int, default=5000,
                        help='Interval steps of TRPO iterations.')
    parser.add_argument('--logger-level', type=int, default=logging.INFO,
                        help='Level of the root logger.')
    parser.add_argument('--use-clipped-gaussian', action='store_true',
                        help='Use ClippedGaussian instead of Gaussian')
    parser.add_argument('--n-hidden-channels', type=int, default=64,
                        help='Number of hidden channels.')
    parser.add_argument('--label', type=str, default='')
    args = parser.parse_args()

    logging.basicConfig(level=args.logger_level)

    # Set random seed
    chainerrl.misc.set_random_seed(args.seed, gpus=(args.gpu,))

    args.outdir = chainerrl.experiments.prepare_output_dir(args, args.outdir)

    def make_env(test):
        env = gym.make(args.env)
        env_seed = 2 ** 32 - 1 - args.seed if test else args.seed
        assert 0 <= env_seed < 2 ** 32
        env.seed(env_seed)
        mode = 'evaluation' if test else 'training'
        env = gym.wrappers.Monitor(
            env,
            args.outdir,
            mode=mode,
            video_callable=False,
            uid=mode,
        )
        if args.render:
            env = CallRender(env)
        env = ClipAction(env)
        return env

    env = make_env(test=False)
    timestep_limit = env.spec.tags.get(
        'wrapper_config.TimeLimit.max_episode_steps')
    obs_space = env.observation_space
    action_space = env.action_space
    print('Observation space:', obs_space)
    print('Action space:', action_space, action_space.low, action_space.high)

    if not isinstance(obs_space, gym.spaces.Box):
        print("""\
This example only supports gym.spaces.Box observation spaces. To apply it to
other observation spaces, use a custom phi function that convert an observation
to numpy.ndarray of numpy.float32.""")  # NOQA
        return

    # Parameterize log std
    def var_func(x): return F.exp(x) ** 2

    # Normalize observations based on their empirical mean and variance
    obs_normalizer = chainerrl.links.EmpiricalNormalization(
        obs_space.low.size)

    assert isinstance(action_space, gym.spaces.Box)
    # Use a Gaussian policy for continuous action spaces
    if args.use_clipped_gaussian:
        policy = \
            ClippedGaussianPolicy(
                obs_space.low.size,
                action_space.low.size,
                n_hidden_channels=args.n_hidden_channels,
                n_hidden_layers=2,
                mean_wscale=0.01,
                nonlinearity=F.tanh,
                var_type='diagonal',
                var_func=var_func,
                var_param_init=0,  # log std = 0 => std = 1
                min_action=action_space.low.astype(np.float32),
                max_action=action_space.high.astype(np.float32),
            )
    else:
        policy = \
            chainerrl.policies.FCGaussianPolicyWithStateIndependentCovariance(
                obs_space.low.size,
                action_space.low.size,
                n_hidden_channels=args.n_hidden_channels,
                n_hidden_layers=2,
                mean_wscale=0.01,
                nonlinearity=F.tanh,
                var_type='diagonal',
                var_func=var_func,
                var_param_init=0,  # log std = 0 => std = 1
            )

    # Use a value function to reduce variance
    vf = chainerrl.v_functions.FCVFunction(
        obs_space.low.size,
        n_hidden_channels=args.n_hidden_channels,
        n_hidden_layers=2,
        last_wscale=0.01,
        nonlinearity=F.tanh,
    )

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        policy.to_gpu(args.gpu)
        vf.to_gpu(args.gpu)
        obs_normalizer.to_gpu(args.gpu)

    # TRPO's policy is optimized via CG and line search, so it doesn't require
    # a chainer.Optimizer. Only the value function needs it.
    vf_opt = chainer.optimizers.Adam()
    vf_opt.setup(vf)

    # Draw the computational graph and save it in the output directory.
    fake_obs = chainer.Variable(
        policy.xp.zeros_like(obs_space.low, dtype=np.float32)[None],
        name='observation')
    chainerrl.misc.draw_computational_graph(
        [policy(fake_obs)], os.path.join(args.outdir, 'policy'))
    chainerrl.misc.draw_computational_graph(
        [vf(fake_obs)], os.path.join(args.outdir, 'vf'))

    # Hyperparameters in http://arxiv.org/abs/1709.06560
    agent = chainerrl.agents.TRPO(
        policy=policy,
        vf=vf,
        vf_optimizer=vf_opt,
        obs_normalizer=obs_normalizer,
        phi=lambda x: x.astype(np.float32, copy=False),
        update_interval=args.trpo_update_interval,
        conjugate_gradient_max_iter=20,
        conjugate_gradient_damping=1e-1,
        gamma=0.995,
        lambd=0.97,
        vf_epochs=5,
        entropy_coef=0,
    )

    if args.load:
        agent.load(args.load)

    if args.demo:
        env = make_env(test=True)
        eval_stats = chainerrl.experiments.eval_performance(
            env=env,
            agent=agent,
            n_runs=args.eval_n_runs,
            max_episode_len=timestep_limit)
        print('n_runs: {} mean: {} median: {} stdev {}'.format(
            args.eval_n_runs, eval_stats['mean'], eval_stats['median'],
            eval_stats['stdev']))
    else:

        chainerrl.experiments.train_agent_with_evaluation(
            agent=agent,
            env=env,
            eval_env=make_env(test=True),
            outdir=args.outdir,
            steps=args.steps,
            eval_n_runs=args.eval_n_runs,
            eval_interval=args.eval_interval,
            max_episode_len=timestep_limit,
            save_best_so_far_agent=False,
        )


if __name__ == '__main__':
    main()
