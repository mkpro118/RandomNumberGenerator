from dataclasses import dataclass
from functools import wraps
from time import perf_counter
from typing import (
    Any, Callable, Generator, List, Optional, Sequence, Union,
)
import math


ListOrNumber = Union[List[float], List[int], float, int]


__all__ = (
    'RNG',
)


class RNG:
    def __init__(self, seed: Optional[int] = None, *,
                 inc: Optional[int] = None):
        self._seed = seed or RNG.INTERNALS._get_seed_()
        self._inc = inc or RNG.INTERNALS.MAGIC3

    ############################################################################
    #          Necessary Constants and Functions for Pseudo-Random RNG         #
    ############################################################################

    class INTERNALS:
        MAGIC1 = 0X41C64E6D  # 1103515245
        MAGIC2 = 0X2C9277B5  # 747796405
        MAGIC3 = 0X25EEE6B6  # 636413622
        MAGIC4 = 0x7FFFFFFF  # 2147483647

        @staticmethod
        def _get_seed_() -> int:
            pc = perf_counter()
            pc = pc - int(pc)
            pc = int(str(pc)[2:])
            return (pc * RNG.INTERNALS.MAGIC1) & RNG.INTERNALS.MAGIC4

    ############################################################################
    #                     Bookkeeping Classes and Functions                    #
    ############################################################################

    @dataclass
    class RNGState:
        seed: Callable[[], int]
        inc: Callable[[], int]

        def validate(self):
            assert all((
                isinstance(self.seed, int),
                isinstance(self.inc, int),
            )), 'Invalid RNGState'

    def get_state(self) -> RNGState:
        state = RNG.RNGState(
            lambda: self._seed,
            lambda: self._inc,
        )

        try:
            state.validate()
        except AssertionError:
            raise ValueError(f'RNG instance {self} has invalid internal state')

        return state

    def set_state(self, state: RNGState):
        assert isinstance(
            self, RNG.RNGState), (
            f'Invalid state type, required \'RNG.RNGState\' found {type(state)}'
        )

        state.validate()
        self._seed = state.seed()
        self._inc = state.inc()

    def seed(self, value: int):
        assert value >= 1, 'seed must be a positive integer'
        self._seed = int(value)

    ############################################################################
    #                            Main RNG Generator                            #
    ############################################################################

    def _generator(self, size: int = 1) -> Generator[float, None, None]:
        assert size >= 1, 'size must be a positive integer'

        if not isinstance(size, int):
            size = int(size)

        for _ in range(size):
            self._seed *= RNG.INTERNALS.MAGIC2
            self._seed += self._inc
            self._seed &= RNG.INTERNALS.MAGIC4
            yield self._seed / RNG.INTERNALS.MAGIC4

    ############################################################################
    #                         Real-Valued Distributions                        #
    ############################################################################

    def random(self) -> float:
        return next(self._generator())

    def uniform(self, size: int = 1,
                low: float = 0.0,
                high: float = 1.0) -> ListOrNumber:
        assert size >= 1, 'size must be a positive integer'

        if high < low:
            low, high = high, low

        res = map(lambda x: low + ((high - low) * x), self._generator(size))

        if size == 1:
            return next(res)

        return list(res)

    def bernoulli(self, p_success: float = 0.5,
                  size: int = 1) -> ListOrNumber:
        assert size >= 1, 'size must be a positive integer'
        assert 0. <= p_success <= 1., 'p_success must be in range [0, 1]'

        if size == 1:
            return int(next(self._generator()) <= p_success)

        return list(map(lambda x: int(x <= p_success), self._generator(size)))

    def binomial(self, n_trials: int = 100,
                 p_success: float = 0.5,
                 size: int = 1) -> ListOrNumber:
        assert size >= 1, 'size must be a positive integer'
        assert 0. <= p_success <= 1., 'p_success must be in range [0, 1]'
        assert n_trials >= 1, 'n_trials must be a positive integer'

        res = [float(sum(map(lambda x: int(x <= p_success),
                             self._generator(size=n_trials)))) for _ in range(size)]

        if size == 1:
            return float(res[-1])

        return res

    def poisson(self, lambd: float = 1.,
                size: int = 1) -> ListOrNumber:
        assert size >= 1, 'size must be a positive integer'
        assert lambd > 0, 'lambd must be a positive real value'

        def _poisson():
            rand_gen = self._generator(size=max(50 * lambd, 10))
            x, p = 0, 1
            while p >= math.exp(-lambd):
                p *= next(rand_gen)
                x += 1
            return x

        if size == 1:
            return _poisson()

        return [_poisson() for _ in range(size)]

    def exponential(self, lambd: float = 1.,
                    size: int = 1) -> ListOrNumber:
        assert size >= 1, 'size must be a positive integer'
        assert lambd > 0, 'lambd must be a positive real value'

        def inverse_transform(x: float) -> float:
            return -(1 / lambd) * math.log(1 - x)

        if size == 1:
            return inverse_transform(next(self._generator()))

        return list(map(inverse_transform, self._generator(size=size)))

    def normal(self, mean: float = 0., stddev: float = 1.,
               size: int = 1) -> ListOrNumber:

        def scale(func: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(func)
            def inner(*args, **kwargs) -> float:
                res = func(*args, **kwargs)
                return res * stddev + mean
            return inner

        # Decorator implementation has been used to visually reflect the
        # function computing the true Box-Muller Transform, as scaling is not a
        # part of the transform
        # Slightly computationally expensive, but visually beautiful
        @scale
        def box_muller_transform(x: tuple[float, float]) -> float:
            return math.sqrt(-2 * math.log(x[0])) * math.cos(2 * math.pi * x[1])

        if size == 1:
            return box_muller_transform(self.random(), self.random())

        rand_gen1 = self._generator(size=size)
        rand_gen2 = self._generator(size=size)

        return list(map(box_muller_transform, zip(rand_gen1, rand_gen2)))

    ############################################################################
    #                          Functions for Integers                          #
    ############################################################################

    def randrange(self, bound1: int, bound2: Optional[int] = None,
                  step: Optional[int] = None):
        if step is not None:
            assert bound1 is not None and bound2 is not None, (
                'cannot have steps without known bounds'
            )
            bound1, bound2 = sorted((bound1, bound2))
            return self.choice(list(range(bound1, bound2, step)))

        if bound2 is not None:
            bound1, bound2 = sorted((bound1, bound2))
            return self.choice(list(range(bound1, bound2)))

        return self.choice(list(range(bound1)))

    def randint(self, a: int, b: Optional[int] = None) -> int:
        if b is None:
            return self.randrange(0, a + 1)

        return self.randrange(a, b + 1)

    ############################################################################
    #                          Functions for Sequences                         #
    ############################################################################

    def choice(self, sequence: Sequence[Any],
               n_samples: int = 1,
               repeat: bool = False) -> Union[Any, List[Any]]:
        assert isinstance(n_samples, int), 'n_samples needs to be an integer'
        assert n_samples >= 1, 'cannot choose lesser than 1 samples'

        if n_samples == 1:
            return sequence[self.randrange(len(sequence))]

        if repeat:
            return [sequence[self.randrange(len(sequence))]
                    for _ in range(n_samples)]

        assert n_samples <= len(sequence), (
            f'cannot choose {n_samples} from '
            f'{len(sequence)} elements without repetition. '
            f'Use `repeat=True` to allow repetition in choices'
        )

        seq_copy = list(sequence)

        return [seq_copy.pop(self.randrange(len(seq_copy))) for _ in range(n_samples)]

    def shuffle(self, sequence: List) -> List:
        if not isinstance(sequence, List):
            sequence = list(sequence)

        def _shuffle(seq: List) -> List:
            if len(seq) == 1:
                return seq

            if len(seq) == 2:
                if self.random() > 0.5:
                    seq[0], seq[1] = seq[1], seq[0]
                return seq

            left, right = _shuffle(seq[::2]), _shuffle(seq[1::2])

            idx = 0
            while len(left) and len(right):
                seq[idx] = left.pop() if self.random() > 0.5 else right.pop()
                idx += 1

            while len(left):
                seq[idx] = left.pop(self.randrange(len(left)))
                idx += 1

            while len(right):
                seq[idx] = right.pop(self.randrange(len(right)))
                idx += 1

            return seq

        return _shuffle(sequence)


del ListOrNumber
