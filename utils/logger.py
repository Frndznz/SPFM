from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import torch
import wandb
os.environ["WANDB_MODE"]="dryrun"
from utils.io_utils import write_args, save_config_to_yaml


class Logger(object):
    def __init__(self, args, config):
        self.args = args
        self.save_dir = args.save_dir
        self.config = config
        self.enable_wandb = config['solver']['logger']['enable_wandb']
        os.makedirs(self.save_dir, exist_ok=True)     
        log_dir = os.path.join(self.save_dir, 'logs')

        # save the args and config, instantiate wandb logger
        if self.enable_wandb:
            self.wandb_logger = wandb.init(
                    project=config['solver']['logger']['project'],
                    dir=log_dir,
                    entity=config['solver']['logger']['entity'],
                    config=config
            )
        else:
            self.wandb_logger = None
        self.config_dir = os.path.join(self.save_dir, 'configs')
        os.makedirs(self.config_dir, exist_ok=True)
        file_name = os.path.join(self.config_dir, 'args.txt')
        write_args(args, file_name)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True, mode=0o777)
        self.text_writer = open(os.path.join(log_dir, 'log.txt'), 'a') # 'w')
        self.tb_writer = None
    
    def log_metrics(self, metrics_dict, step=None):
        """Logs a dictionary of metrics to wandb."""
        if self.wandb_logger:
            if step is not None:
                self.wandb_logger.log(metrics_dict, step=step)
            else:
                self.wandb_logger.log(metrics_dict)

    def save_config(self, config):
        save_config_to_yaml(config, os.path.join(self.config_dir, 'config.yaml'))

    def log_info(self, info, check_primary=True):
        print(info)
        info = str(info)
        time_str = time.strftime('%Y-%m-%d-%H-%M')
        info = '{}: {}'.format(time_str, info)
        if not info.endswith('\n'):
            info += '\n'
        self.text_writer.write(info)
        self.text_writer.flush()

    def add_scalar(self, **kargs):
        """Log a scalar variable."""
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(**kargs)

    def add_scalars(self, **kargs):
        """Log a scalar variable."""
        if self.tb_writer is not None:
            self.tb_writer.add_scalars(**kargs)

    def add_image(self, **kargs):
        """Log a scalar variable."""
        if self.tb_writer is not None:
            self.tb_writer.add_image(**kargs)

    def add_images(self, **kargs):
        """Log a scalar variable."""
        if self.tb_writer is not None:
            self.tb_writer.add_images(**kargs)

    def close(self):
        self.text_writer.close()
        self.tb_writer.close()

