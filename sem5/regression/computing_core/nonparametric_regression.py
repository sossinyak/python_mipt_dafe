from typing import Union
import numpy as np

from enum import Enum


class ComutitionTypes(Enum):
    PYTHON = 'python'
    NUMPY = 'numpy'


class NonparametricRegression:

    __X: Union[None, np.ndarray] = None
    __y: Union[None, np.ndarray]
    __k_neighbours: int = 5
    __computition_type: str = ComutitionTypes.PYTHON

    def __init__(
            self, 
            k_neighbours: int = 5, 
            comp_type=ComutitionTypes.PYTHON
        ) -> None:

        if comp_type not in [ComutitionTypes.PYTHON, ComutitionTypes.NUMPY]:
            message = f'invalid comp_type value: {comp_type}; '
            message += 'only python and numpy are allowed;'
            raise RuntimeError(message)

        if k_neighbours <= 0:
            message = f'invalid k_neighbours value: {k_neighbours}; '
            message += 'positive integer numbers are required;'
            raise RuntimeError(message)

        self.__k_neighbours = int(k_neighbours)
        self.__computition_type = comp_type

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        
        if not isinstance(X, np.ndarray):
            message = f'Unexpected X type: {type(X).__name__}; '
            message += 'numpy.ndarray was expected;'
            raise RuntimeError(message)
        
        if not isinstance(y, np.ndarray):
            message = f'Unexpected y type: {type(y).__name__}; '
            message += 'numpy.ndarray was expected;'
            raise RuntimeError(message)
        
        self.__X = np.array(X, dtype=np.float64)
        self.__y = np.array(y, dtype=np.float64)

    def predict(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        
        if self.__X is None or self.__y is None:
            message = 'fit() method should be called before;'
            raise RuntimeError(message)
        
        weights = None
        
        if self.__computition_type == ComutitionTypes.PYTHON:
            weights = self.__compute_weights_python(x)

        elif self.__computition_type == ComutitionTypes.NUMPY:
            weights = self.__compute_weights_numpy(x)

        pass

    def __compute_weights_python(self, x: float) -> list:

        distances = list(abs(xi - x) for xi in self.__X)
        window_size = sorted(distances)[self.__k_neighbours - 1]
        distances_normalized = [dist / window_size for dist in distances]

        kernel = lambda xi: 0.75 * (1 - xi ** 2) if xi < 1 else 0
        weights = list(map(kernel, distances_normalized))

        return weights
