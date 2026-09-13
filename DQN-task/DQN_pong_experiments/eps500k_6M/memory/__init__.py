from memory.memory import ReplayBuffer

BUFFER_DICT = {
    "ReplayBuffer": ReplayBuffer,
    "PrioritizedReplayBuffer": None, # you can implement PrioritizedReplayBuffer by yourself and add it to this dict
}