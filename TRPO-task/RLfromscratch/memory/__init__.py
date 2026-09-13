from memory.memory import ReplayBuffer, TrajectoryRollout

BUFFER_DICT = {
    "ReplayBuffer": ReplayBuffer,
    "TrajectoryRollout": TrajectoryRollout,
    "PrioritizedReplayBuffer": None, # you can implement PrioritizedReplayBuffer by yourself and add it to this dict
}