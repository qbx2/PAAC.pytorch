import argparse
import random

import gym
import torch
import torch.nn
from torch.autograd import Variable

from paac import PAACNet, INPUT_CHANNELS, INPUT_IMAGE_SIZE
from worker import Worker
from logger import Logger


class Master:
    def __init__(self, args):
        self.args = args

        print('Loading environment information ...')
        env = gym.make(args.env)

        self.num_actions = env.action_space.n
        print('num_actions:', self.num_actions)

        self.action_meanings = env.env.get_action_meanings()
        print('action_meanings:', self.action_meanings)

        self.no_op = None

        for i, v in enumerate(self.action_meanings):
            if v.upper() == 'NOOP':
                self.no_op = i
                print('Using action %d as NO-OP' % i)

        if self.no_op is None:
            self.no_op = 0
            print('NO-OP not found, using action 0')

        del env

        # create PAAC model
        self.paac = PAACNet(self.num_actions)

        if args.cuda:
            self.paac.cuda()

        if args.use_rmsprop:
            self.optim = torch.optim.RMSprop(
                self.paac.parameters(), args.learning_rate, args.alpha,
                args.rmsprop_epsilon
            )
        else:
            self.optim = torch.optim.Adam(
                self.paac.parameters(), args.learning_rate,
                (args.beta1, args.beta2), args.adam_epsilon
            )

        self.workers = [Worker(i, args) for i in range(args.n_w)]
        self.start = 0
        self.range_iter = None

    @staticmethod
    def get_starting_point():
        return random.randint(args.min_starting_point, args.max_starting_point)

    @staticmethod
    def normalize_reward(reward):
        if reward < -1:
            return -1
        elif reward > 1:
            return 1

        return reward

    def train(self):
        optim = self.optim
        workers = self.workers
        args = self.args
        model_params = self.paac.parameters()
        (
            filename, cuda, n_e, t_max, n_max,
            gamma, beta, log_step, save_step,
            epsilon, clip
        ) = (
            args.filename, args.cuda, args.n_e, args.t_max, args.n_max,
            args.gamma, args.beta, args.log_step, args.save_step,
            args.epsilon, args.clip
        )
        log_step_1 = log_step - 1
        save_step_1 = save_step - 1
        del args

        # gpu (if possible) variables, will be wrapped by Variable later
        # policies = Variable(torch.zeros(t_max, n_e))  # unused at the moment
        values = torch.zeros(t_max, n_e)
        log_a = torch.zeros(t_max, n_e)

        # gpu tensors
        # tensor to store states, updated at every timestep
        states = torch.zeros(n_e, INPUT_CHANNELS, *INPUT_IMAGE_SIZE)
        # tensors to store data for a backprop
        rewards = torch.zeros(t_max, n_e)
        terminals = torch.zeros(t_max, n_e)
        q_values = torch.zeros(t_max + 1, n_e)

        # cpu tensors
        # selected actions by policy network to be sent to workers
        # actions = [0] * n_e
        # if current_frames < starting_points: action = no-op
        # else: action = policy()
        starting_points = [self.get_starting_point() for _ in range(n_e)]
        current_frames = [1] * n_e
        # accumulated rewards to calculate score
        rewards_accumulated = [0] * n_e
        normalized_rewards_accumulated = [0] * n_e
        # list to store scores of episodes,
        # printed & flushed at every log_step
        scores = []
        normalized_scores = []
        # sum of loss_p & double_loss_v, printed & flushed at every log_step
        loss_p_sum = double_loss_v_sum = entropy_sum = 0

        if cuda:
            # policies = policies.pin_memory().cuda(async=True)
            values = values.pin_memory().cuda(async=True)
            log_a = log_a.pin_memory().cuda(async=True)

            states = states.pin_memory().cuda(async=True)
            rewards = rewards.pin_memory().cuda(async=True)
            terminals = terminals.pin_memory().cuda(async=True)
            q_values = q_values.pin_memory().cuda(async=True)

        # wrap variables
        # policies = Variable(policies)
        values = Variable(values)
        log_a = Variable(log_a)

        # start training
        self.paac.train()

        # init states
        for worker in workers:
            obs, *_ = worker.get_observations()

            for i, ob in enumerate(obs, worker.env_start):
                states[i, -1] = ob

        self.range_iter = iter(range(self.start, n_max))

        for n in self.range_iter:
            # policies = Variable(policies.data)
            values = Variable(values.data)
            log_a = Variable(log_a.data)

            negated_entropy_sum = 0

            for t in range(t_max):
                # check terminals[t_max - 1] when t = 0
                for i, terminal in enumerate(terminals[t - 1]):
                    if terminal:
                        # reset done environments
                        starting_points[i] = self.get_starting_point()
                        current_frames[i] = 1

                        scores.append(rewards_accumulated[i])
                        normalized_scores.append(
                            normalized_rewards_accumulated[i])

                        rewards_accumulated[i] = 0
                        normalized_rewards_accumulated[i] = 0

                        states[i].fill_(0)

                # states must be cloned for gradient calculation
                paac_p, paac_v = self.paac(Variable(states.clone()))
                paac_p_max_values, paac_p_max_indices = paac_p.max(1)
                values[t] = paac_v

                log_paac_p, negated_h = self.paac.log_and_negated_entropy(
                        paac_p, epsilon)
                negated_entropy_sum += negated_h

                actions = list(paac_p_max_indices.data)

                # process no-op environments
                for i in range(n_e):
                    if current_frames[i] < starting_points[i]:
                        current_frames[i] += 1
                        # policies[t, i] = paac_p[i, self.NOOP]
                        actions[i] = self.no_op

                    log_a[t, i] = log_paac_p[i, actions[i]]

                # perform actions
                for worker in workers:
                    worker.put_actions(actions[worker.env_slice])

                # get new observations
                for worker in workers:
                    obs = worker.get_observations()

                    for i, (ob, reward, done) in \
                            enumerate(zip(*obs), worker.env_start):
                        states[i, :-1], states[i, -1] = states[i, 1:], ob
                        rewards[t, i] = normalized_rewards = \
                            self.normalize_reward(reward)
                        terminals[t, i] = done

                        rewards_accumulated[i] += reward
                        normalized_rewards_accumulated[i] += normalized_rewards

            entropy = -negated_entropy_sum / n_e
            entropy_sum += entropy.data[0]

            # values of new states
            q_values[t_max] = self.paac.value(Variable(states)).data

            loss_sum = 0

            # calculate q_values
            for t in reversed(range(t_max)):
                q_values[t] = rewards[t] + \
                              (1. - terminals[t]) * gamma * q_values[t + 1]

                loss_p, double_loss_v, loss = self.paac.get_loss(
                    q_values[t], values[t], log_a[t]
                )

                loss_sum += loss
                loss_p_sum += loss_p.data[0]
                double_loss_v_sum += double_loss_v.data[0]

            # entropy term
            loss_sum -= beta * entropy

            optim.zero_grad()
            # loss scaling by t_max
            loss_sum.backward()
            torch.nn.utils.clip_grad_norm(model_params, clip)
            optim.step()

            if n % log_step == log_step_1:
                Logger.log(**locals())

                # flush
                loss_p_sum = double_loss_v_sum = entropy_sum = 0
                scores.clear()
                normalized_scores.clear()

                if n % save_step == save_step_1:
                    self.save(filename, n + 1)

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.start = checkpoint['iteration']
        self.paac.load_state_dict(checkpoint['paac'])
        self.optim.load_state_dict(checkpoint['optimizer'])
        print('Loaded PAAC checkpoint (%d) from' % self.start, filename)

    def save(self, filename, iteration=0):
        checkpoint = {
            'iteration': iteration,
            'paac': self.paac.state_dict(),
            'optimizer': self.optim.state_dict()
        }

        torch.save(checkpoint, filename)
        print('Saved PAAC checkpoint (%d) into' % iteration, filename)


def get_args():
    parser = argparse.ArgumentParser(
        description='Train a PAAC model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--env', type=str, default='Pong-v0')
    parser.add_argument('-f', '--filename', type=str, default='paac.pkl',
                        help='filename to save the trained model into.')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('-l', '--log-step', type=int, default=100)
    parser.add_argument('-s', '--save-step', type=int, default=1000)
    # WARNING: you should check if the agent can control the environment
    # in the starting point range (e. g. The agent cannot control
    # until 35th frame in SpaceInvadersDeterministic-v4)
    parser.add_argument('--min-starting-point', type=int, default=1)
    parser.add_argument('--max-starting-point', type=int, default=30)
    # crayon experiment name
    parser.add_argument('--experiment-name', type=str, default='paac')

    # PAAC parameters
    parser.add_argument('-w', '--n_w', '--workers', type=int,
                        default=8, metavar='N_W',
                        help='Number of workers')
    parser.add_argument('-e', '--n_e', '--environments', type=int,
                        default=32, metavar='N_E',
                        help='Number of environments')
    parser.add_argument('-t', '--t-max', type=int, default=5, metavar='T_MAX',
                        help='Max local steps')

    parser.add_argument('-n', '--n-max', type=int, default=int(1.15e8),
                        metavar='N_MAX',
                        help='Max global steps')
    parser.add_argument('-g', '--gamma', type=float, default=0.99)

    # Optimizer parameters
    parser.add_argument('--lr', '--learning-rate', type=float, default=0.00224,
                        dest='learning_rate', help='Learning rate')
    parser.add_argument('--use-adam', dest='use_rmsprop', action='store_false')

    # RMSProp parameters
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='Alpha for the RMSProp optimizer')
    parser.add_argument('--rmsprop-epsilon', type=float, default=0.1,
                        help='Epsilon for the RMSProp optimizer')

    # Adam parameters
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Beta1 for the Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Beta2 for the Adam optimizer')
    parser.add_argument('--adam-epsilon', type=float, default=1e-8)

    # Other parameters
    parser.add_argument('-b', '--beta', type=float, default=0.01,
                        help='Strength of entropy regularization term')
    parser.add_argument('-E', '--epsilon', type=float, default=1e-30,
                        help='Epsilon for numerical stability')
    parser.add_argument('-C', '--clip', type=float, default=40.0)

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    return args


if __name__ == '__main__':
    args = get_args()
    print(args)

    Logger.set_experiment_name(args.experiment_name)

    master = Master(args)

    try:
        master.load(args.filename)
    except FileNotFoundError as e:
        print(e)

    try:
        master.train()
    finally:
        try:
            n = next(master.range_iter)
        except TypeError:
            n = master.start

        master.save(args.filename, n)
