import wandb
import datetime


class Logger:
    step = 0

    def __init__(self, config, logger_name='logger', project='inm706'):
        logger_name = f'{logger_name}-{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
        logger = wandb.init(project=project, name=logger_name, config=config)
        self.logger = logger

    def log(self, data):
        self.logger.log(data)

    def watch(self, model):
        self.logger.watch(model, log='all')

    def log_artifact(self, artifact):
        self.logger.log_artifact(artifact)
