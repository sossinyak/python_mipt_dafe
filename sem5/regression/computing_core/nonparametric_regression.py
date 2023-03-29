from collections.abc import Iterable
from numbers import Real
from typing import Union
from enum import StrEnum

import numpy as np


class ComutitionTypes(StrEnum):
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

    def fit(
            self, X: Iterable, 
            y: Iterable
        ) -> None:
        
        if not isinstance(X, Iterable):
            message = f'incorrect X type: {type(X).__name__}; '
            message += 'iterable object was expected'
            raise TypeError(message)
        
        if not isinstance(y, Iterable):
            message = f'incorrect y type: {type(y).__name__}; '
            message += 'iterable object was expected'
            raise TypeError(message)

        self.__X = np.array(X, dtype=np.float64)
        self.__y = np.array(y, dtype=np.float64)

    def predict(self, x: Union[Real, Iterable]) -> Union[Real, Iterable]:
        
        if self.__X is None or self.__y is None:
            message = 'fit() method should be called before;'
            raise RuntimeError(message)
        
        if self.__computition_type == ComutitionTypes.PYTHON:

            if isinstance(x, Real):
                weights = self.__compute_weights_python(x)
                return self.__compute_result_python(weights)

            else:
                message = f'incorrect x type: {type(x).__name__}; '
                message += 'real number was expected;'
                raise RuntimeError(message)

        elif self.__computition_type == ComutitionTypes.NUMPY:

            if isinstance(x, Iterable):
                x = np.array(x, dtype=np.float64)
                weights = self.__compute_weights_numpy(x)

                return self.__compute_result_numpy(weights)

            else:
                message = f'incorrect x type: {type(x).__name__}; '
                message += 'Iterable object was expected;'
                raise RuntimeError(message)

    def __compute_weights_python(self, x: float) -> list:

        distances = list(abs(xi - x) for xi in self.__X)
        window_size = sorted(distances)[self.__k_neighbours - 1]
        distances_normalized = [dist / window_size for dist in distances]

        kernel = lambda xi: 0.75 * (1 - xi ** 2) if xi < 1 else 0
        weights = list(map(kernel, distances_normalized))

        return weights
    
    def __compute_weights_numpy(self, X: np.ndarray) -> np.ndarray:

        distances = np.abs(X[:, np.newaxis] - self.__X)
        window_sizes = np.sort(distances)[:, self.__k_neighbours - 1]
        distances_normalized = distances / window_sizes[:, np.newaxis]

        weights = np.where(
            distances_normalized < 1,
            0.75 * (1 - distances_normalized ** 2),
            np.zeros_like(distances_normalized)
        )

        return weights
    
    def __compute_result_python(self, weights: list) -> float:

        numerator = sum(wi * yi for wi, yi in zip(weights, self.__y))
        denominator = sum(weights)

        return numerator / denominator
    
    def __compute_result_numpy(self, weights: np.ndarray) -> np.ndarray:

        numerator = np.sum(weights * self.__y, axis=1)
        denominator = np.sum(weights, axis=1)

        return numerator / denominator
