import neptune

from abc import ABC, abstractmethod
from typing import Optional, Any
from nip import nip


class AbstractLogger(ABC):

    @abstractmethod
    def init(self):
        raise NotImplementedError()

    @abstractmethod
    def log(self, key: str, value: Any, desc: Optional[str] = None):
        raise NotImplementedError()

    @abstractmethod
    def close(self):
        raise NotImplementedError()


@nip
class ConsoleLogger(AbstractLogger):

    def init(self):
        pass

    def log(self, key: str, value: Any, desc: Optional[str] = None):
        print_str = key
        if desc is not None:
            print_str = f"{print_str} ({desc})"
        print_str = f"{print_str}:    {value}"
        print(print_str)

    def close(self):
        pass


@nip
class NeptuneLogger(AbstractLogger):

    def __init__(self, neptune_project: str):
        super().__init__()
        self._neptune_project = neptune_project
        self._run = None

    def init(self):
        if self._run is None:
            self._run = neptune.init_run(project=self._neptune_project)

    def log(self, key: str, value: Any, desc: Optional[str] = None):
        self._run[key].append(value)

    def close(self):
        self._run.stop()
