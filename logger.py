class Logger:
    exp = None

    @staticmethod
    def log(n, t_max, n_e, log_step,
            loss_p_sum, double_loss_v_sum, entropy_sum,
            scores, normalized_scores, **kwargs):
        iteration, timestep = n + 1, (n + 1) * t_max * n_e
        print('Iteration %d (Timestep %d)' % (iteration, timestep))

        average_loss_p = loss_p_sum / log_step / t_max
        average_loss_v = double_loss_v_sum / 2. / log_step / t_max
        average_entropy = entropy_sum / log_step / t_max

        print('average loss_p:', average_loss_p)
        print('average loss_v:', average_loss_v)
        print('average entropy:', average_entropy)

        print('Episodes:', len(scores))

        try:
            max_score = max(scores)
            min_score = min(scores)
            avg_score = sum(scores) / len(scores)

            print('Max_score:', max_score)
            print('Min_score:', min_score)
            print('Avg_score:', avg_score)
        except ValueError:
            pass

        try:
            max_norm_score = max(normalized_scores)
            min_norm_score = min(normalized_scores)
            avg_norm_score = sum(normalized_scores) / len(normalized_scores)

            print('Max_norm_score:', max_norm_score)
            print('Min_norm_score:', min_norm_score)
            print('Avg_norm_score:', avg_norm_score)
        except ValueError:
            pass

        print()

        if Logger.exp is not None:
            import requests

            try:
                Logger.crayon_log(**locals())
            except requests.ConnectionError as e:
                print(e)

    @staticmethod
    def init_crayon(hostname, experiment_name):
        try:
            from pycrayon import CrayonClient

            cc = CrayonClient(hostname)

            try:
                Logger.exp = cc.create_experiment(experiment_name)
            except ValueError as e:
                print(e)

                if input('Open the experiment (y/n)? ').lower() != 'y':
                    raise

                Logger.exp = cc.open_experiment(experiment_name)
        except ImportError:
            print('Importing pycrayon has been failed. '
                  'Some features of Logger will disabled.')
        except ValueError as e:
            print(e)

            if input('continue (y/n)? ').lower() != 'y':
                raise

    @staticmethod
    def crayon_log(timestep, average_loss_p, average_loss_v, average_entropy,
                   max_score=None, min_score=None, avg_score=None,
                   max_norm_score=None, min_norm_score=None,
                   avg_norm_score=None, **kwargs):
        exp = Logger.exp
        exp.add_scalar_value("loss_p", average_loss_p, step=timestep)
        exp.add_scalar_value("loss_v", average_loss_v, step=timestep)
        exp.add_scalar_value("entropy", average_entropy, step=timestep)

        if max_score is not None:
            exp.add_scalar_value("score_max", max_score, step=timestep)

        if min_score is not None:
            exp.add_scalar_value("score_min", min_score, step=timestep)

        if avg_score is not None:
            exp.add_scalar_value("score_avg", avg_score, step=timestep)

        if max_norm_score is not None:
            exp.add_scalar_value("norm_score_max", max_norm_score,
                                 step=timestep)

        if min_norm_score is not None:
            exp.add_scalar_value("norm_score_min", min_norm_score,
                                 step=timestep)

        if avg_norm_score is not None:
            exp.add_scalar_value("norm_score_avg", avg_norm_score,
                                 step=timestep)
