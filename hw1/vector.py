from numbers import Integral, Real
from itertools import zip_longest
from functools import reduce
from array import array


class Vector:

    __components: array
    __repr_lim: int = 5

    def __init__(self, iterable):

        # ВАШ КОД
        self.__components = array('d', iterable)

    def __repr__(self):

        #ВАШ КОД
        return 'Vector repr'

    def __str__(self):

        # ВАШ КОД
        return 'Vector str'

    def __iter__(self):
        return iter(self.__components)

    def __len__(self):

        # ВАШ КОД
        return 0

    def __abs__(self):

        # ВАШ КОД
        return 0

    def __bool__(self):

        # ВАШ КОД
        return False

    def __getitem__(self, index):

        # ВАШ КОД
        return self.__components[index]

    def __setitem__(self, index, value):

        # ВАШ КОД
        self.__components[index] = value

    def __eq__(self, other):

        # ВАШ КОД
        return False

    def __neg__(self):
        
        # ВАШ КОД
        return Vector([])

    def __add__(self, other):

        # ВАШ КОД
        return Vector([])

    def __radd__(self, other):
        
        # ВАШ КОД
        return Vector([])

    def __sub__(self, other):

        # ВАШ КОД
        return Vector([])

    def __mul__(self, scalar):

        # ВАШ КОД
        return Vector([])

    def __rmul__(self, scalar):

        # ВАШ КОД
        return Vector([])

    def __matmul__(self, other):

        # ВАШ КОД
        return 0

    def __rmatmul__(self, other):

        # ВАШ КОД
        return 0

    def __truediv__(self, scalar):

        # ВАШ КОД
        return Vector([])
