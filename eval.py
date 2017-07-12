import argparse

import gym
import torch
from torch.autograd import Variable

from paac import PAACNet, INPUT_CHANNELS, INPUT_IMAGE_SIZE


def get_args():
    parser = argparse.ArgumentParser(
        description='Evaluate a PAAC model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--env', type=str, default='Pong-v0')
    parser.add_argument('-f', '--filename', type=str, default='paac.pkl',
                        help='filename to save the trained model into.')
    parser.add_argument('-c', '--cuda', type=bool,
                        default=torch.cuda.is_available())
    parser.add_argument('-d', '--debug', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    env = gym.make(args.env)
    env.env._get_obs = env.env.ale.getScreenGrayscale
    ob = env.reset()

    num_actions = env.action_space.n
    print('num_actions:', num_actions)

    action_meanings = env.env.get_action_meanings()
    print('action_meanings:', action_meanings)

    paac = PAACNet(num_actions)
    state_dict = torch.load(args.filename)

    try:
        iteration = state_dict['_iteration']
        del state_dict['_iteration']
    except KeyError:
        iteration = -1

    paac.load_state_dict(state_dict)
    print('Loaded PAAC model (%d) from' % iteration, args.filename)

    paac.eval()

    state = torch.zeros(1, INPUT_CHANNELS, *INPUT_IMAGE_SIZE)
    score = 0

    while True:
        state[0, :-1] = state[0, 1:]
        state[0, -1] = PAACNet.preprocess(ob)
        env.render()

        policy, value = paac(Variable(state, volatile=True))
        action = policy.max(1)[1].data[0]

        if args.debug:
            print(action_meanings[action])

        ob, reward, done, info = env.step(action)
        score += reward

        if done:
            print('score:', score)
            score = 0
            state.fill_(0)
            ob = env.reset()
