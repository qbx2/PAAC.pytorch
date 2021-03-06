import argparse

import torch
from torch.autograd import Variable

import gym_wrapper as gym
from paac import PAACNet, INPUT_CHANNELS, INPUT_IMAGE_SIZE


def get_args():
    parser = argparse.ArgumentParser(
        description='Evaluate a PAAC model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--env', type=str, default='Pong-v0')
    parser.add_argument('-f', '--filename', type=str, default='paac.pkl',
                        help='filename to save the trained model into.')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('--use-max', action='store_true')

    return parser.parse_args()


def draw_state(state):
    import matplotlib.pyplot as plt
    plt.ion()
    for i in range(4):
        plt.subplot(141 + i)
        plt.imshow(PAACNet.to_pil_image(state[:, i]), cmap='gray')
    plt.pause(1e-30)


if __name__ == '__main__':
    args = get_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda

    env = gym.make(args.env, hack='eval')
    ob = env.reset()

    num_actions = env.action_space.n
    print('num_actions:', num_actions)

    action_meanings = env.env.get_action_meanings()
    print('action_meanings:', action_meanings)

    paac = PAACNet(num_actions)
    checkpoint = torch.load(args.filename,
                            map_location=lambda storage, loc: storage)

    try:
        iteration = checkpoint['iteration']
    except KeyError:
        iteration = -1

    paac.load_state_dict(checkpoint['paac'])
    print('Loaded PAAC checkpoint (%d) from' % iteration, args.filename)

    paac.eval()

    state = torch.zeros(1, INPUT_CHANNELS, *INPUT_IMAGE_SIZE)
    score = 0

    if args.cuda:
        paac.cuda()
        state = state.pin_memory().cuda(async=True)

    while True:
        state[0, :-1] = state[0, 1:]
        state[0, -1] = PAACNet.preprocess(ob)
        env.render()

        # draw_state(state)

        policy, value = paac(Variable(state, volatile=True))

        if args.use_max:
            action = policy.max(1)[1].cpu().data[0]
        else:
            action = policy.multinomial()[0].cpu().data[0]

        if args.debug:
            entropy = paac.entropy(policy, 1e-30)

            print('policy:', policy.data.numpy())
            print('value:', value.data[0, 0])
            print('entropy:', entropy.data[0])
            print(action_meanings[action])

        ob, reward, done, info = env.step(action)
        score += reward

        if done:
            print('score:', score)
            score = 0
            state.fill_(0)
            ob = env.reset()
