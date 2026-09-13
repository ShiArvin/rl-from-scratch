import pickle
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def _load_data(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.suffix == ".pkl":
        with path.open("rb") as f:
            data = pickle.load(f)
    elif path.suffix == ".npz":
        data = dict(np.load(path, allow_pickle=True))
    else:
        raise ValueError(f"Unsupported expert dataset format: {path.suffix}")

    if not isinstance(data, dict):
        raise ValueError("Expert dataset must be a dict-like object with states and actions.")
    return data


def _load_minari_data(
    dataset_id: str,
    download: bool = False,
    datasets_path: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if datasets_path:
        os.environ["MINARI_DATASETS_PATH"] = str(datasets_path)

    try:
        import minari
    except ImportError as exc:
        raise ImportError("Minari expert dataset loading requires the minari package.") from exc

    dataset = minari.load_dataset(dataset_id, download=download)
    states = []
    actions = []
    for episode in dataset.iterate_episodes():
        episode_states = np.asarray(episode.observations)
        episode_actions = np.asarray(episode.actions)
        if len(episode_states) == len(episode_actions) + 1:
            episode_states = episode_states[:-1]
        elif len(episode_states) != len(episode_actions):
            raise ValueError(
                "Minari episode observations/actions length mismatch: "
                f"{len(episode_states)} observations vs {len(episode_actions)} actions."
            )

        states.append(episode_states)
        actions.append(episode_actions)

    if len(states) == 0:
        raise ValueError(f"Minari dataset {dataset_id!r} contains no episodes.")
    return np.concatenate(states, axis=0), np.concatenate(actions, axis=0)


def _flatten_array_or_trajectories(value: Any, name: str) -> np.ndarray:
    if isinstance(value, list):
        if len(value) == 0:
            raise ValueError(f"Expert dataset field {name} is empty.")
        return np.concatenate([np.asarray(traj) for traj in value], axis=0)

    array = np.asarray(value)
    if array.dtype == object:
        items = [np.asarray(item) for item in array.tolist()]
        if len(items) == 0:
            raise ValueError(f"Expert dataset field {name} is empty.")
        return np.concatenate(items, axis=0)
    return array


def _flatten_paired_expert_data(states: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if actions.ndim >= 3 and states.shape[: actions.ndim - 1] == actions.shape[:-1]:
        leading_ndim = actions.ndim - 1
        states = states.reshape((-1, *states.shape[leading_ndim:]))
        actions = actions.reshape((-1, actions.shape[-1]))
    elif actions.ndim >= 2 and states.ndim > actions.ndim and states.shape[: actions.ndim] == actions.shape:
        leading_ndim = actions.ndim
        states = states.reshape((-1, *states.shape[leading_ndim:]))
        actions = actions.reshape((-1,))
    return states, actions


def load_expert_data(
    path: str | Path,
    dataset_format: str = "minari",
    state_key: str = "states",
    action_key: str = "actions",
    minari_download: bool = False,
    minari_datasets_path: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    dataset_format = dataset_format.lower()
    if dataset_format == "auto":
        suffix = Path(path).suffix
        dataset_format = suffix[1:] if suffix in {".pkl", ".npz"} else "minari"

    if dataset_format == "minari":
        return _load_minari_data(
            str(path),
            download=minari_download,
            datasets_path=minari_datasets_path,
        )
    if dataset_format not in {"pkl", "npz"}:
        raise ValueError(f"Unsupported expert dataset format: {dataset_format}")

    data = _load_data(path)
    if state_key not in data or action_key not in data:
        raise KeyError(f"Expert dataset must contain {state_key!r} and {action_key!r}.")

    states = _flatten_array_or_trajectories(data[state_key], state_key)
    actions = _flatten_array_or_trajectories(data[action_key], action_key)
    states, actions = _flatten_paired_expert_data(states, actions)

    if len(states) != len(actions):
        raise ValueError(
            f"Expert states/actions length mismatch: {len(states)} states vs {len(actions)} actions."
        )
    return states, actions


class ExpertDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        dataset_format: str = "minari",
        state_key: str = "states",
        action_key: str = "actions",
        minari_download: bool = False,
        minari_datasets_path: str | None = None,
    ):
        states, actions = load_expert_data(
            path,
            dataset_format=dataset_format,
            state_key=state_key,
            action_key=action_key,
            minari_download=minari_download,
            minari_datasets_path=minari_datasets_path,
        )
        self.states = states.astype(np.float32, copy=False)
        self.actions = actions.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "states": torch.as_tensor(self.states[idx], dtype=torch.float32),
            "actions": torch.as_tensor(self.actions[idx], dtype=torch.float32),
        }
