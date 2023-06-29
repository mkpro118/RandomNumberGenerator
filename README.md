# RNG

RNG is a Python library that provides functions for generating random numbers and performing random operations. It offers a simple interface for generating random values from various distributions and manipulating sequences randomly. The library is built upon the RNG(Random Number Generator) class, providing a flexible and customizable random number generation process.

# Installation

You can get RNG using `git clone`:

```bash
git clone https://github.com/mkpro118/RandomNumberGenerator.git
```

# Usage

Here's an example of how to use RNG:

```python
import rng

# Create an instance of the RNG class
rng = rng.default_rng()

# Generate a random float between 0 and 1
value = rng.random()

# Generate a random integer between 1 and 10
rand_int = rng.randint(1, 10)

# Generate a list of 5 random floats between -1 and 1
rand_floats = rng.uniform(size=5, low=-1, high=1)

# Choose a random element from a sequence
numbers = [1, 2, 3, 4, 5]
random_choice = rng.choice(numbers)

# Shuffle a list randomly
rng.shuffle(numbers)

# Generate a list of 10 random integers from a binomial distribution
binomial_samples = rng.binomial(n_trials=10, p_success=0.5, size=10)
```

# Contributing

Contributions to RNG are welcome! If you encounter any issues, have suggestions for improvements, or would like to add new features, please feel free to open an issue or submit a pull request on the GitHub repository.

# License

RNG is licensed under the MIT License. See the LICENSE file for more information.
