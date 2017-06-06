from abc import ABCMeta, abstractmethod
from typing import TypeVar, Generic, Callable
import numpy as np

from toolz.curried import curry, compose, map, pipe, pluck

A = TypeVar('A')
B = TypeVar('B')

class Functor(Generic[A], metaclass=ABCMeta):
    @abstractmethod
    def fmap(self, func: Callable[[A], B]) -> 'Functor[B]':
        return NotImplemented

    def __repr__(self) -> str:
        return self.__str__()

class Maybe(Generic[A], Functor[A], metaclass=ABCMeta):
    pass

class Just(Generic[A], Maybe[A]):
    def __init__(self, value: A) -> None:
        self._value = value

    def fmap(self, func: Callable[[A], B]) -> 'Just[B]':
        return Just(func(self._value))

    def __str__(self) -> str:
        return 'Just({value})'.format(value=self._value)

class Nothing(Generic[A], Maybe[A]):
    def fmap(self, func: Callable[[A], B]) -> Maybe[B]:
        return Nothing()

    def __str__(self) -> str:
        return 'Nothing'

class IO(Generic[A], Functor[A]):
    def __init__(self, action):
        self._action = action

    def fmap(self, func: Callable[[A], B]) -> 'IO[B]':
        return IO(lambda: func(self._action()))

    def __call__(self, *args, **kwargs) -> 'IO[A]':
        return IO(lambda: self._action(*args, **kwargs))

    def __str__(self) -> str:
        return 'IO({value})'.format(value=self._action())

class Random(Generic[A], Functor[A]):
    def __init__(self, action):
        self._action = action

    def fmap(self, func: Callable[[A], B]) -> 'Random[B]':
        return Random(lambda: func(self._action()))

    def __call__(self, *args, **kwargs) -> 'Random[A]':
        return Random(lambda: self._action(*args, **kwargs))

    def __str__(self) -> str:
        return 'Random({value})'.format(value=self._action())

@curry
def fmap(func: Callable[[A], B], item: Functor[A]) -> Functor[B]:
    return item.fmap(func)


makeio = IO
makerandom = Random

@curry
def array_from_tuple(data, shape, dtype):
    arr = np.zeros(shape, dtype=dtype)
    for slice_, value in data:
        arr[slice_] = value
    return arr
