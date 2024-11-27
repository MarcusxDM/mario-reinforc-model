from stable_baselines3.common.callbacks import BaseCallback
import os

class TrainAndLoggingCallback(BaseCallback, ):

    def __init__(self, check_freq, save_path, model_name, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.model_name = model_name

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, '{}_{}'.format(self.model_name, self.n_calls))
            self.model.save(model_path)

        return True