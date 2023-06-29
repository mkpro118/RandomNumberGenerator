'''Module: rng

This module provides a set of functions for generating random numbers and
performing random operations.

We provide convenient access to various random number generation and random
operation functions through the default_rng instance and shortcut functions.

To instead create customizable RNG instances, use
```python
from rng import RNG

rand_gen = RNG(seed=5, inc=10)

print(rand_gen.random())     # 0.7410991879837118
print(rand_gen.randint(10))  # 3
print(rand_gen.normal())     # 0.27511590773502925

```


Functions:
1. default_rng(seed: Optional[int] = None) -> RNG:
   - Creates and returns an instance of the RNG class with the specified seed.

2. random() -> float:
   - Returns a random float between 0 and 1, following a uniform distribution.

3. uniform(size: int = 1, low: float = 0.0, high: float = 1.0) -> ListOrNumber:
   - Returns random floats between the specified low and high values,
     following a uniform distribution.

4. normal(mean: float = 0.0, stddev: float = 1.0, size: int = 1) -> ListOrNumber:
   - Returns random floats from a normal distribution with the specified
     mean and standard deviation.

5. poisson(lambd: float = 1.0, size: int = 1) -> ListOrNumber:
   - Returns random integers from a Poisson distribution with the specified
     lambda parameter.

6. binomial(n_trials: int = 100, p_success: float = 0.5, size: int = 1) -> ListOrNumber:
   - Returns random integers from a binomial distribution with the specified
     number of trials and success probability.

7. bernoulli(p_success: float = 0.5, size: int = 1) -> ListOrNumber:
   - Returns random integers from a Bernoulli distribution with the specified
     success probability.

8. exponential(lambd: float = 1.0, size: int = 1) -> ListOrNumber:
   - Returns random floats from an exponential distribution with the specified
     lambda parameter.

9. randint(a: int, b: Optional[int] = None) -> int:
   - Returns a random integer between a and b (inclusive) if b is specified,
     or between 0 and a (inclusive) if b is not specified.

10. randrange(bound1: int, bound2: Optional[int] = None, step: Optional[int] = None) -> int:
    - Returns a randomly selected integer from the specified range.

11. choice(sequence: Sequence[Any], n_samples: int = 1, repeat: bool = False) -> Union[Any, List[Any]]:
    - Returns randomly chosen elements from a sequence.

12. shuffle(sequence: List) -> List:
    - Randomly shuffles the elements of a list in-place.


Usage:
```python
import rng

rng = rng.default_rng(42)
value = rng.random()

numbers = [1, 2, 3, 4, 5]
random_sample = rng.choice(numbers, n_samples=3, repeat=True)

rng.shuffle(numbers)
```

'''

from typing import Optional

from .rng import RNG


def default_rng(seed: Optional[int] = None):
    return RNG(seed)


shortcut_rng = default_rng()

# Set RNG State
seed = shortcut_rng.seed

# Real-Valued Distributions
random = shortcut_rng.random
uniform = shortcut_rng.uniform
normal = shortcut_rng.normal
poisson = shortcut_rng.poisson
binomial = shortcut_rng.binomial
bernoulli = shortcut_rng.bernoulli
exponential = shortcut_rng.exponential

# Integer Functions
randint = shortcut_rng.randint
randrange = shortcut_rng.randrange

# Sequence Functions
choice = shortcut_rng.choice
shuffle = shortcut_rng.shuffle

del shortcut_rng
