from torch.utils.data import Dataset


class ReplayMemory(Dataset):
    def __init__(self, capacity):
        self._capacity = capacity
        self._memory = []
        self._position = 0

    def push(self, item):
        # replace old experience with new experience
        if len(self._memory) < self._position + 1:
            self._memory.append(item)
        else:
            self._memory[self._position] = item
        self._position = (self._position + 1) % self._capacity

    def is_full(self):
        return len(self._memory) == self._capacity

    def __getitem__(self, item):
        return self._memory[item]

    def __len__(self):
        return len(self._memory)

    def clear(self):
        self._memory = []
