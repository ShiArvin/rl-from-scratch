from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from agent.agent import AgentBase
from logger.logger import Logger
from utils.evaluate import evaluate
from utils.result import Result
from utils.utils import get_action_dim


class OfflineAlgorithm(ABC):
    def __init__(
        self,
        training_envs,
        dataset,
        agent: AgentBase,
        logger: Logger,
        device,
        save_pth: str,
        best_pth: str,
        args,
    ):
        self.training_envs = training_envs
        self.dataset = dataset
        self.agent = agent.to(device)
        self.logger = logger
        self.device = device
        self.save_pth = save_pth
        self.best_pth = best_pth
        self.args = args

        self.observation_space = training_envs.observation_space
        self.action_space = training_envs.action_space
        self.action_dim = get_action_dim(self.action_space)
        self.seed = args.seed

        self.interaction_step = 0
        self.gradient_step = 0
        self.best_score = -np.inf

        algo_args = args.algorithm
        self.total_epoch = getattr(algo_args, "total_epoch", args.total_epoch)
        self.test_interval = args.test_interval
        self.save_interval = args.save_interval
        self.train_log_interval = args.train_log_interval
        self.test_episodes = args.test_episodes

        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=algo_args.batch_size,
            shuffle=getattr(algo_args, "shuffle", True),
            drop_last=getattr(algo_args, "drop_last", False),
            num_workers=getattr(algo_args, "num_workers", 0),
        )

    @abstractmethod
    def update_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        pass

    def test_condition(self, epoch: int) -> bool:
        return self.test_interval > 0 and epoch % self.test_interval == 0

    def save_condition(self, epoch: int) -> bool:
        return self.save_interval > 0 and epoch % self.save_interval == 0

    def train_log_condition(self, epoch: int) -> bool:
        return self.train_log_interval > 0 and epoch % self.train_log_interval == 0

    def update_epoch(self) -> Result:
        self.agent.train()
        metric_sums = defaultdict(float)
        metric_count = 0

        with Result("train") as result:
            for batch in self.dataloader:
                metrics = self.update_batch(batch)
                for key, value in metrics.items():
                    metric_sums[key] += float(value)
                metric_count += 1
                self.gradient_step += 1

        if metric_count == 0:
            raise ValueError("Expert dataset is empty; cannot run offline training.")

        for key, value in metric_sums.items():
            result.add_metric(key, value / metric_count)
        return result

    def run(self):
        for epoch in tqdm(range(self.total_epoch), desc="Epoch", unit="epoch"):
            train_result = self.update_epoch()
            if self.train_log_condition(epoch):
                self.logger.log_train(epoch, self.gradient_step, self.gradient_step, train_result)

            if self.test_condition(epoch):
                test_result, test_reward = evaluate(
                    self.agent,
                    self.test_episodes,
                    self.args.env,
                    self.seed + self.gradient_step,
                    self.training_envs,
                )
                self.logger.log_test(epoch, self.interaction_step, self.gradient_step, test_result)
                if test_reward > self.best_score:
                    self.best_score = test_reward
                    self.agent.save(self.best_pth)

            if self.save_condition(epoch):
                self.agent.save(self.save_pth)
