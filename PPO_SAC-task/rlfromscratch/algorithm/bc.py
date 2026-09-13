import gymnasium as gym
import torch

from agent.agent import PolicyAgent
from logger.logger import Logger
from utils import OPTIMIZER_DICT
from .offline import OfflineAlgorithm


class BehaviorCloning(OfflineAlgorithm):
    def __init__(
        self,
        training_envs,
        dataset,
        agent: PolicyAgent,
        logger: Logger,
        device,
        save_pth: str,
        best_pth: str,
        args,
    ):
        if not isinstance(training_envs.action_space, gym.spaces.Box):
            raise NotImplementedError("BehaviorCloning currently implements only continuous Box action spaces.")

        super(BehaviorCloning, self).__init__(
            training_envs=training_envs,
            dataset=dataset,
            agent=agent,
            logger=logger,
            device=device,
            save_pth=save_pth,
            best_pth=best_pth,
            args=args,
        )

        algo_args = args.algorithm
        self.actor_lr = algo_args.actor_lr
        self.entropy_coef = getattr(algo_args, "entropy_coef", 0.0)
        self.use_grad_clip = getattr(algo_args, "use_grad_clip", False)
        self.max_grad_norm = getattr(algo_args, "max_grad_norm", 0.5)
        self.action_eps = getattr(algo_args, "action_eps", 1e-6)

        self.optimizer = OPTIMIZER_DICT[algo_args.optimizer](self.agent.actor.parameters(), lr=self.actor_lr)
        self.action_low = torch.as_tensor(training_envs.action_space.low, device=device, dtype=torch.float32)
        self.action_high = torch.as_tensor(training_envs.action_space.high, device=device, dtype=torch.float32)

    def _prepare_actions(self, actions: torch.Tensor) -> torch.Tensor:
        actions = actions.to(self.device, dtype=torch.float32)
        if actions.ndim == 1 and self.action_dim == 1:
            actions = actions.unsqueeze(1)
        if actions.shape[-1] != self.action_dim:
            raise ValueError(f"Expected action dim {self.action_dim}, got action shape {tuple(actions.shape)}.")
        return torch.clamp(actions, min=self.action_low + self.action_eps, max=self.action_high - self.action_eps)

    def _dist_entropy(self, dist) -> torch.Tensor:
        if hasattr(dist, "base_dist"):
            return dist.base_dist.entropy().mean()
        return dist.entropy().mean()

    def _deterministic_action(self, dist) -> torch.Tensor:
        actor = self.agent.actor
        if actor.rescale:
            if actor.action_bound_method == "tanh":
                pred_actions = actor.scale * torch.tanh(dist.base_dist.mean) + actor.loc
            else:
                raise NotImplementedError
        else:
            pred_actions = dist.base_dist.mean
        return actor._action_clamp(pred_actions)

    def update_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        states = batch["states"].to(self.device, dtype=torch.float32)
        actions = self._prepare_actions(batch["actions"])

        dist = self.agent.dist(states)
        log_probs = dist.log_prob(actions)
        nll_loss = -log_probs.mean()
        entropy = self._dist_entropy(dist)
        loss = nll_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = None
        if self.use_grad_clip:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(), self.max_grad_norm)
        self.optimizer.step()

        with torch.no_grad():
            pred_actions = self._deterministic_action(dist)
            action_mse = torch.mean((pred_actions - actions) ** 2)

        metrics = {
            "actor/loss": loss.item(),
            "actor/nll_loss": nll_loss.item(),
            "actor/entropy": entropy.item(),
            "actor/action_mse": action_mse.item(),
            "actor/log_prob_mean": log_probs.mean().item(),
        }
        if grad_norm is not None:
            metrics["actor/grad_norm"] = float(grad_norm)
        return metrics
