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
    """
    Random Number Generator (RNG) class that provides various methods
    to generate pseudo-random numbers.

    Attributes:
        _seed (int): The seed value for the RNG.
        _inc (int): The increment value for the RNG.

    Methods:
        Bookkeeping functions
            get_state: Retrieves the current state of the RNG.
            set_state: Sets the state of the RNG.
            seed: Sets the seed value for the RNG.

        Real-Valued Distributions
            random: Generates a random number between 0 and 1.
            uniform: Generates random numbers from a uniform distribution.
            bernoulli: Generates random numbers from a Bernoulli distribution.
            binomial: Generates random numbers from a binomial distribution.
            poisson: Generates random numbers from a Poisson distribution.
            exponential: Generates random numbers from an exponential distribution.
            normal: Generates random numbers from a normal distribution.

        Functions for Integers
            randrange: Generates a random integer within a given range.
                       with an exclusive upper bound
            randint: Generates a random integer between two values
                     with an inclusive upper bound

        Functions for Sequences
            choice: Chooses one or more elements randomly from a sequence.
            shuffle: Shuffles the elements in a list randomly.
    """

    def __init__(self, seed: Optional[int] = None, *,
                 inc: Optional[int] = None):
        """
        Initializes the RNG instance.

        Args:
            seed (int): The seed value for the RNG.
                        If not provided, a default seed value is used.
            inc (int): The increment value for the RNG.
                       If not provided, a default value is used.
        """
        self._seed = seed or RNG.INTERNALS._get_seed_()
        self._inc = inc or RNG.INTERNALS.MAGIC3

    ############################################################################
    #          Necessary Constants and Functions for Pseudo-Random RNG         #
    ############################################################################

    class INTERNALS:
        """
        Internal constants and functions for the RNG.

        Note: These are internal constants that are essential to the random
        number generator. Do not modify these unless you know what you're doing!

        Attributes:
            MAGIC1 (int): A magic number used in the RNG algorithm.
                          Value: 1103515245.
            MAGIC2 (int): Another magic number used in the RNG algorithm.
                          Value: 747796405.
            MAGIC3 (int): Yet another magic number used in the RNG algorithm.
                          Value: 636413622.
            MAGIC4 (int): A magic number used for bit masking in the RNG algorithm.
                          Value: 2147483647.
        """

        MAGIC1 = 0X41C64E6D  # 1103515245
        MAGIC2 = 0X2C9277B5  # 747796405
        MAGIC3 = 0X25EEE6B6  # 636413622
        MAGIC4 = 0x7FFFFFFF  # 2147483647

        @staticmethod
        def _get_seed_() -> int:
            """
            Generates a seed value for the RNG based on the current time using
            time.perf_counter()

            The fractional part of the value returned by time.perf_counter() is
            converted to an integer, which is then reduced to a value between
            0 and 2 ** 31 - 1

            Returns:
                int: The generated seed value.
            """
            pc = perf_counter()
            pc = pc - int(pc)
            pc = int(str(pc)[2:])
            return (pc * RNG.INTERNALS.MAGIC1) & RNG.INTERNALS.MAGIC4

    ############################################################################
    #                     Bookkeeping Classes and Functions                    #
    ############################################################################

    @dataclass
    class RNGState:
        """
        Represents the state of an RNG instance.

        Attributes:
            seed: A function that returns the current seed value of the RNG.
            inc: A function that returns the current increment value of the RNG.

        Methods:
            validate(): Validates the integrity of the RNGState instance.

        Raises:
            AssertionError: If the RNGState instance is invalid.

        Note:
            The RNGState class is intended for internal use within the RNG class
            and is used to represent the state of the RNG instance for reproducibility.
        """
        seed: Callable[[], int]
        inc: Callable[[], int]

        def validate(self):
            assert all((
                isinstance(self.seed, int),
                isinstance(self.inc, int),
            )), 'Invalid RNGState'

    def get_state(self) -> RNGState:
        """
        Retrieves the current state of the RNG instance.

        Returns:
            RNGState: The current state of the RNG instance.

        Raises:
            ValueError: If the RNG instance has an invalid internal state.
        """
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
        """
        Sets the state of the RNG instance to the provided state.

        Args:
            state (RNGState): The state to set for the RNG instance.

        Raises:
            AssertionError: If the provided state is not of type RNGState.
            ValueError: If the provided state is invalid.
        """
        assert isinstance(
            self, RNG.RNGState), (
            f'Invalid state type, required \'RNG.RNGState\' found {type(state)}'
        )

        state.validate()
        self._seed = state.seed()
        self._inc = state.inc()

    def seed(self, value: int):
        """
        Sets the seed value for the RNG instance.

        Args:
            value (int): The seed value to set.

        Raises:
            AssertionError: If the provided seed value is not a positive integer.
        """
        assert value >= 1, 'seed must be a positive integer'
        self._seed = int(value)

    ############################################################################
    #                            Main RNG Generator                            #
    ############################################################################

    def _generator(self, size: int = 1) -> Generator[float, None, None]:
        """
        Internal generator function that yields random floating-point
        numbers between 0 and 1.

        Args:
            size (int, optional): The number of random numbers to generate.
                                  Defaults to 1.

        Yields:
            float: A random floating-point number between 0 and 1.

        Raises:
            AssertionError: If the provided size is not a positive integer.
        """
        assert size >= 1, 'size must be a positive integer'

        if not isinstance(size, int):
            size = int(size)

        for _ in range(size):
            # Update the seed value using RNG.INTERNALS.MAGIC2 and self._inc
            self._seed *= RNG.INTERNALS.MAGIC2
            self._seed += self._inc
            self._seed &= RNG.INTERNALS.MAGIC4

            # Yield the generated random number between 0 and 1
            yield self._seed / RNG.INTERNALS.MAGIC4

    ############################################################################
    #                         Real-Valued Distributions                        #
    ############################################################################

    def random(self) -> float:
        """
        Generates a single random floating-point number between 0 and 1.

        Returns:
            float: A random floating-point number between 0 and 1.
        """
        return next(self._generator())

    def uniform(self, size: int = 1,
                low: float = 0.0,
                high: float = 1.0) -> ListOrNumber:
        """
        Generates random floating-point numbers uniformly distributed
        between `low` and `high`.

        Args:
            size (int, optional): The number of random numbers to generate.
                                  Defaults to 1.
            low (float, optional): The lower bound of the range.
                                   Defaults to 0.0.
            high (float, optional): The upper bound of the range.
                                   Defaults to 1.0.

        Returns:
            ListOrNumber: A single random number if `size` is 1,
                          otherwise a list of random numbers.

        Raises:
            AssertionError: If the provided size is not a positive integer.

        Notes:
            - If `high` is less than `low`, the values are swapped
              to ensure correct range calculation.
            - The returned values are uniformly distributed between
              `low` and `high`.
        """
        assert size >= 1, 'size must be a positive integer'

        if high < low:
            low, high = high, low

        res = map(lambda x: low + ((high - low) * x), self._generator(size))

        if size == 1:
            return next(res)

        return list(res)

    def bernoulli(self, p_success: float = 0.5,
                  size: int = 1) -> ListOrNumber:
        """
        Generates random binary values based on the Bernoulli distribution.

        Args:
            p_success (float, optional): The probability of success for a
                                         binary outcome. Defaults to 0.5.
            size (int, optional): The number of random binary values to generate.
                                  Defaults to 1.

        Returns:
            ListOrNumber: A single random binary value if `size` is 1,
                          otherwise a list of random binary values.

        Raises:
            AssertionError: If the provided size is not a positive integer.
            AssertionError: If the provided p_success is not in the range [0, 1].

        Notes:
            - The returned binary values follow the Bernoulli distribution
              with the specified probability of success.
            - A binary value of 1 represents success, and 0 represents failure.
        """
        assert size >= 1, 'size must be a positive integer'
        assert 0. <= p_success <= 1., 'p_success must be in range [0, 1]'

        if size == 1:
            return int(next(self._generator()) <= p_success)

        return list(map(lambda x: int(x <= p_success), self._generator(size)))

    def binomial(self, n_trials: int = 100,
                 p_success: float = 0.5,
                 size: int = 1) -> ListOrNumber:
        """
        Generates random numbers based on the binomial distribution.

        Args:
            n_trials (int, optional): The number of trials in each experiment.
                                      Defaults to 100.
            p_success (float, optional): The probability of success for each trial.
                                         Defaults to 0.5.
            size (int, optional): The number of random numbers to generate.
                                  Defaults to 1.

        Returns:
            ListOrNumber: A single random number if `size` is 1,
                          otherwise a list of random numbers.

        Raises:
            AssertionError: If the provided size is not a positive integer.
            AssertionError: If the provided p_success is not in the range [0, 1].
            AssertionError: If the provided n_trials is not a positive integer.

        Notes:
            - The binomial distribution represents the number of successes in a
              fixed number of independent Bernoulli trials.
            - Each random number generated corresponds to the sum of successes
              over `n_trials` trials.
            - The returned random numbers follow the binomial distribution with
              the specified `n_trials` and `p_success`.
        """
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
        """
        Generates random numbers based on the Poisson distribution.

        Args:
            lambd (float, optional): The average rate of the Poisson distribution.
                                     Defaults to 1.0.
            size (int, optional): The number of random numbers to generate.
                                  Defaults to 1.

        Returns:
            ListOrNumber: A single random number if `size` is 1,
                          otherwise a list of random numbers.

        Raises:
            AssertionError: If the provided size is not a positive integer.
            AssertionError: If the provided lambd is not a positive real value.

        Notes:
            - The Poisson distribution represents the number of events occurring
              in a fixed interval of time or space.
            - Each random number generated corresponds to the number of events
              occurring based on the Poisson distribution.
            - The returned random numbers follow the Poisson distribution with
              the specified average rate `lambd`.
        """
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
        """
        Generates random numbers based on the exponential distribution.

        Args:
            lambd (float, optional): The rate parameter of the exponential
                                     distribution. Defaults to 1.0.
            size (int, optional): The number of random numbers to generate.
                                  Defaults to 1.

        Returns:
            ListOrNumber: A single random number if `size` is 1,
                          otherwise a list of random numbers.

        Raises:
            AssertionError: If the provided size is not a positive integer.
            AssertionError: If the provided lambd is not a positive real value.

        Notes:
            - The exponential distribution models the time between events in a
              Poisson process.
            - Each random number generated corresponds to the time between
              consecutive events based on the exponential distribution.
            - The returned random numbers follow the exponential distribution
              with the specified rate parameter `lambd`.
        """
        assert size >= 1, 'size must be a positive integer'
        assert lambd > 0, 'lambd must be a positive real value'

        def inverse_transform(x: float) -> float:
            return -(1 / lambd) * math.log(1 - x)

        if size == 1:
            return inverse_transform(next(self._generator()))

        return list(map(inverse_transform, self._generator(size=size)))

    def normal(self, mean: float = 0., stddev: float = 1.,
               size: int = 1) -> ListOrNumber:
        """
        Generates random numbers based on the normal (Gaussian) distribution.

        Args:
            mean (float, optional): The mean of the normal distribution.
                                    Defaults to 0.0.
            stddev (float, optional): The standard deviation of the normal
                                      distribution. Defaults to 1.0.
            size (int, optional): The number of random numbers to generate.
                                  Defaults to 1.

        Returns:
            ListOrNumber: A single random number if `size` is 1,
                          otherwise a list of random numbers.

        Notes:
            - The normal distribution is a continuous probability distribution
              that is symmetric about the mean.
            - The generated random numbers follow the normal distribution with
              the specified `mean` and `stddev`.
            - The Box-Muller transform is used to generate random numbers from
              a standard normal distribution (mean = 0, stddev = 1).
            - The generated standard normal random numbers are then scaled by
              `stddev` and shifted by `mean` to match the desired distribution.
        """

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
            """
            Applies the Box-Muller transform to generate a random number from a
            standard normal distribution.

            Args:
                x (tuple[float, float]): A tuple of two random numbers from a
                                         uniform distribution (0, 1).

            Returns:
                float: A random number from a standard normal distribution.

            Notes:
                - The Box-Muller transform converts two independent random
                  numbers from a uniform distribution to a standard normal
                  random number.
            """
            return math.sqrt(-2 * math.log(x[0])) * math.cos(2 * math.pi * x[1])

        if size == 1:
            return box_muller_transform((self.random(), self.random()))

        rand_gen1 = self._generator(size=size)
        rand_gen2 = self._generator(size=size)

        return list(map(box_muller_transform, zip(rand_gen1, rand_gen2)))

    ############################################################################
    #                          Functions for Integers                          #
    ############################################################################

    def randrange(self, bound1: int, bound2: Optional[int] = None,
                  step: Optional[int] = None):
        """
        Return a randomly selected element from the range created by the arguments.

        Args:
            bound1 (int): The start (or stop if `bound2` is None) of the range.
            bound2 (int, optional): The stop of the range. If None, `bound1` is considered as the stop and 0 is considered as the start. Defaults to None.
            step (int, optional): The step size between the elements in the range. If provided, `bound1` and `bound2` must be known. Defaults to None.

        Returns:
            int: A randomly selected integer from the generated range.

        Raises:
            AssertionError: If `step` is provided but `bound1` or `bound2` is None.

        Notes:.
            - The generated range includes `bound1` but excludes `bound2`
              (or vice versa if `bound1` is greater than `bound2`).
            - If `step` is provided, the method selects a random element from
              the range based on the step size.
            - If both `bound1` and `bound2` are provided, the method sorts them
              in ascending order to ensure the range is correctly generated.
        """
        if step is not None:
            assert step != 0, 'step cannot be 0'
            assert bound1 is not None and bound2 is not None, (
                'cannot have steps without known bounds'
            )
            if step > 0:
                bound1, bound2 = sorted((bound1, bound2))
            return self.choice(list(range(bound1, bound2, step)))

        if bound2 is not None:
            bound1, bound2 = sorted((bound1, bound2))
            return self.choice(list(range(bound1, bound2)))

        return self.choice(list(range(bound1)))

    def randint(self, a: int, b: Optional[int] = None) -> int:
        """
        Return a random integer between `a` and `b` (inclusive)
        if `b` is provided, or between 0 and `a` (inclusive) otherwise.

        Args:
            a (int): The lower or upper bound of the range,
                     depending on the presence of `b`.
            b (int, optional): The upper bound of the range.
                               If None, `a` is considered as the upper bound
                               and 0 is considered as the lower bound.
                               Defaults to None.

        Returns:
            int: A randomly selected integer between `a` and `b` (inclusive).

        Notes:
            - The method internally uses the `randrange()` method
              to generate the random integer.
            - If `b` is not provided, the range is generated from
              0 to `a` (inclusive).
        """
        if b is None:
            return self.randrange(0, a + 1)

        return self.randrange(a, b + 1)

    ############################################################################
    #                          Functions for Sequences                         #
    ############################################################################

    def choice(self, sequence: Sequence[Any],
               n_samples: int = 1,
               repeat: bool = False) -> Union[Any, List[Any]]:
        """
        Return a randomly selected element or a list of randomly selected
        elements from the given sequence.

        Args:
            sequence (Sequence): The sequence from which to choose the elements.
            n_samples (int, optional): The number of elements to choose.
                                       Defaults to 1.
            repeat (bool, optional): Whether to allow repetition of elements
                                     in the chosen samples. Defaults to False.

        Returns:
            Union[Any, List]: A randomly selected element if `n_samples` is 1,
                              or a list of randomly selected elements otherwise.

        Raises:
            AssertionError: If `n_samples` is not an integer or is less than 1.
            AssertionError: If `n_samples` is greater than the length of the
                            sequence without allowing repetition.

        Notes:
            - If `n_samples` is 1, a single element is randomly selected
              from the sequence and returned.
            - If `n_samples` is greater than 1 and `repeat` is False, a list of
              distinct elements is randomly selected from the sequence and returned.
            - If `n_samples` is greater than the length of the sequence
              and `repeat` is False, an AssertionError is raised.
            - If `repeat` is True, elements may be repeated in the chosen samples.
        """
        assert isinstance(n_samples, int), 'n_samples needs to be an integer'
        assert n_samples >= 1, 'cannot choose lesser than 1 samples'

        if n_samples == 1:
            return sequence[int(self.uniform(low=0, high=len(sequence) + 1))]

        if repeat:
            return [sequence[int(self.uniform(low=0, high=len(sequence) + 1))]
                    for _ in range(n_samples)]

        assert n_samples <= len(sequence), (
            f'cannot choose {n_samples} from '
            f'{len(sequence)} elements without repetition. '
            f'Use `repeat=True` to allow repetition in choices'
        )

        seq_copy = list(sequence)

        return [seq_copy.pop(
            int(self.uniform(low=0, high=len(seq_copy) + 1))) for _ in range(n_samples)]

    def shuffle(self, sequence: List) -> List:
        """
        Shuffle the elements of the given sequence in place and
        return the shuffled sequence.

        Args:
            sequence (List): The sequence to be shuffled.

        Returns:
            List: The shuffled sequence.

        Notes:
            - A method similar to merge-sort is used to shuffle the sequence
            - The method modifies the input sequence in place.
        """
        if not isinstance(sequence, List):
            # This ensures that `sequence` has a `pop` method
            sequence = list(sequence)

        def _shuffle(seq: List) -> List:
            '''
            This function is basically merge sort, except instead of sorting,
            we're shuffling the array
            '''
            if len(seq) == 1:
                return seq

            if len(seq) == 2:
                if self.random() > 0.5:
                    seq[0], seq[1] = seq[1], seq[0]
                return seq

            # Recursively shuffle
            left, right = _shuffle(seq[::2]), _shuffle(seq[1::2])

            # shuffle while merging
            idx = 0
            while len(left) and len(right):
                seq[idx] = left.pop() if self.random() > 0.5 else right.pop()
                idx += 1

            # add remaining elements from left, if any, continue the shuffling
            while len(left):
                seq[idx] = left.pop(self.randrange(len(left)))
                idx += 1

            # add remaining elements from right, if any, continue the shuffling
            while len(right):
                seq[idx] = right.pop(self.randrange(len(right)))
                idx += 1

            return seq

        return _shuffle(sequence)


del ListOrNumber
