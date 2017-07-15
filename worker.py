from torch.multiprocessing import Process, Pipe, Event

import gym_wrapper as gym
from paac import PAACNet


class Worker(Process):
    def __init__(self, worker_id, args):
        super().__init__()

        self.id = worker_id
        self.args = args
        # for master use, for worker use
        self.pipe_master, self.pipe_worker = Pipe()
        self.exit_event = Event()

        # determine n_e
        q, r = divmod(args.n_e, args.n_w)

        if r:
            print('Warning: n_e % n_w != 0')

        if worker_id == args.n_w - 1:
            self.n_e = n_e = q + r
        else:
            self.n_e = n_e = q

        print('Worker', self.id, '] n_e = %d' % n_e)

        self.env_start = worker_id * q
        self.env_slice = slice(self.env_start, self.env_start + n_e)
        self.env_range = range(self.env_start, self.env_start + n_e)
        self.envs = None

        self.start()

    def make_environments(self):
        envs = []

        for _ in range(self.n_e):
            envs.append(gym.make(self.args.env, hack='train'))

        return envs

    def put_shared_tensors(self, actions, obs, rewards, terminals):
        assert (actions.is_shared() and obs.is_shared() and
                rewards.is_shared() and terminals.is_shared())

        self.pipe_master.send((actions, obs, rewards, terminals))

    def get_shared_tensors(self):
        actions, obs, rewards, terminals = self.pipe_worker.recv()
        assert (actions.is_shared() and obs.is_shared() and
                rewards.is_shared() and terminals.is_shared())
        return actions, obs, rewards, terminals

    def set_step_done(self):
        self.pipe_worker.send_bytes(b'1')

    def wait_step_done(self):
        self.pipe_master.recv_bytes(1)

    def set_action_done(self):
        self.pipe_master.send_bytes(b'1')

    def wait_action_done(self):
        self.pipe_worker.recv_bytes(1)

    def run(self):
        preprocess = PAACNet.preprocess

        envs = self.envs = self.make_environments()
        env_start = self.env_start
        t_max = self.args.t_max
        t = 0
        dones = [False] * self.args.n_e

        # get shared tensor
        actions, obs, rewards, terminals = self.get_shared_tensors()

        for i, env in enumerate(envs, start=env_start):
            obs[i] = preprocess(env.reset())

        self.set_step_done()

        while not self.exit_event.is_set():
            self.wait_action_done()

            for i, env in enumerate(envs, start=env_start):
                if not dones[i]:
                    ob, reward, done, info = env.step(actions[i])
                else:
                    ob, reward, done, info = env.reset(), 0, False, None

                obs[i] = preprocess(ob)
                rewards[t, i] = reward
                terminals[t, i] = dones[i] = done

            self.set_step_done()

            t += 1

            if t == t_max:
                t = 0
