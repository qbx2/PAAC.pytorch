import gym


def make(env_id, hack=None):
    if 'Deterministic-v4' not in env_id:
        print('[Warning] Use Deterministic-v4 version '
              'to reproduce the results of paper.')

    _env = env = gym.make(env_id)

    if hack:
        # Hack gym env to output grayscale image
        if env.spec.timestep_limit is not None:
            from gym.wrappers.time_limit import TimeLimit

            if isinstance(env, TimeLimit):
                _env = env.env

        if hack == 'train':
            _env._get_image = _env.ale.getScreenGrayscale
            _env._get_obs = _env.ale.getScreenGrayscale
        elif hack == 'eval':
            _env._get_obs = _env.ale.getScreenGrayscale

    return env
