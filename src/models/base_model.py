from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def train(self, X: np.ndarray, Y: np.ndarray, **kwargs):
        pass

    @abstractmethod
    def test(self, X: np.ndarray, Y: np.ndarray):
        pass