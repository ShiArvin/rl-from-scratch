
import os 
from typing import Any

from utils.result import Result 
import wandb 
from torch.utils.tensorboard import SummaryWriter
from rich.console import Console 
from rich.table import Table 



def _print(training: bool, epoch: int, interaction_step: int, gradient_step: int, metrics: dict[str, Any]):
    console = Console()
    if training:
        title = "[bold yellow]Training Info[/bold yellow]"
    else:
        title = "[bold green]Testing Info[/bold green]"
    table = Table(title=title)
    table.add_column("Metric", justify="left")
    table.add_column("Value", justify="right")

    metrics['epoch'] = epoch 
    metrics['interaction_step'] = interaction_step
    metrics['gradient_step'] = gradient_step

    for k in sorted(sorted(metrics.keys())):
        table.add_row(k, str(metrics[k]))
    
    console.print(table)



class Logger:
    def __init__(self, project_name:str, run_name:str, log_dir="logs", use_wandb=True, use_tensorboard=True):
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        self.log_dir = log_dir
        if self.use_wandb:
            wandb.init(project=project_name, name=run_name,dir=log_dir)
        
        if self.use_tensorboard:
            self.writer = SummaryWriter(log_dir=log_dir)

    def _log(self, key, value, step):
        if self.use_wandb:
            wandb.log({key: value}, step=step)
        if self.use_tensorboard:
            self.writer.add_scalar(key, value, global_step=step)

    def log_train(self, epoch, interaction_step, gradient_step, train_result: Result): 
        for k,v in train_result.metrics.items():
            if "network" in k:
                self._log(k, v, gradient_step)
            else:
                self._log(f"train/{k}", v, interaction_step) # for non-loss metrics, we log them with interaction steps, for loss metrics, we log them with gradient steps

            
        _print(training=True, epoch=epoch, interaction_step=interaction_step, gradient_step=gradient_step, metrics=train_result.metrics)
            
    def log_test(self, epoch, interaction_step, gradient_step, test_result: Result):
        for k,v in test_result.metrics.items():
            self._log(f"test/{k}", v, interaction_step)
        _print(training=False, epoch=epoch, interaction_step=interaction_step, gradient_step=gradient_step, metrics=test_result.metrics)

    
    def close(self):
        if self.use_wandb:
            wandb.finish()
        if self.use_tensorboard:
            self.writer.close()
