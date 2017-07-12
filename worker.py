from multiprocessing import Process, Pipe
import gym

from paac import PAACNet


class Worker(Process):
    def __init__(self, worker_id, args):
        super().__init__()

        self.id = worker_id
        self.args = args
        self.pipe_master, self.pipe_worker = Pipe()  # master to worker

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
        self.envs = None

        self.start()

    def make_environments(self):
        envs = []

        for _ in range(self.n_e):
            # Hack gym env to output grayscale image
            env = gym.make(self.args.env)
            env.env._get_image = env.env.ale.getScreenGrayscale
            env.env._get_obs = env.env.ale.getScreenGrayscale
            envs.append(env)

        return envs

    def put_states(self, states):
        self.pipe_worker.send(states)

    def get_observations(self):
        return self.pipe_master.recv()

    def put_actions(self, actions):
        self.pipe_master.send(actions)

    def get_actions(self):
        return self.pipe_worker.recv()

    def run(self):
        preprocess = PAACNet.preprocess
        envs = self.envs = self.make_environments()
        obs = [preprocess(env.reset()) for env in envs]
        rewards = [0] * self.n_e
        dones = [False] * self.n_e

        while True:
            self.put_states((obs, rewards, dones))
            actions = self.get_actions()

            for i, (env, action) in enumerate(zip(envs, actions)):
                if dones[i]:
                    obs[i] = preprocess(env.reset())
                    rewards[i] = 0
                    dones[i] = False
                else:
                    ob, rewards[i], dones[i], info = env.step(action)
                    obs[i] = preprocess(ob)
