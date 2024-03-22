import wandb
import datetime


class Logger:

    def __init__(self, config, logger_name='logger', project='inm706'):
        logger_name = f'{logger_name}-{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
        logger = wandb.init(project=project, name=logger_name, config=config)
        self.logger = logger
        return

    def get_logger(self):
        return self.logger
