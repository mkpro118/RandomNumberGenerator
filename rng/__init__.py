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
