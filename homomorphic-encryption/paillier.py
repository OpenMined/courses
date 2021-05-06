from collections import namedtuple
import secrets
from typing import List, Tuple

import numpy as np  # type: ignore

PrivateKey = namedtuple("PrivateKey", ["lam", "mu"])
PublicKey = namedtuple("PublicKey", ["g", "n", "n_squared"])

DEFAULT_BIT_LENGTH = 3072

# Copy and paste your function from the Cryptography lesson or uncomment this code:
# def generate_primes(n: int) -> List[int]:
#     # https://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n-in-python/3035188#3035188
#     """ Input n>=6, Returns an array of primes, 2 <= p < n """
#     sieve = np.ones(n // 3 + (n % 6 == 2), dtype=bool)
#     for i in range(1, int(n ** 0.5) // 3 + 1):
#         if sieve[i]:
#             k = 3 * i + 1 | 1
#             sieve[k * k // 3 :: 2 * k] = False
#             sieve[k * (k - 2 * (i & 1) + 4) // 3 :: 2 * k] = False
#     primes = np.r_[2, 3, ((3 * np.nonzero(sieve)[0][1:] + 1) | 1)]
#     return [int(n) for n in primes]


def L(n: int, x: int) -> int:
    pass


def create_key_pair(
    # Used to specify the desired bit length of the modulus n
    bit_length: int = DEFAULT_BIT_LENGTH,
) -> Tuple[PrivateKey, PublicKey]:
    pass


def encrypt(public_key: PublicKey, plaintext: int) -> int:
    pass


def decrypt(private_key: PrivateKey, public_key: PublicKey, ciphertext: int) -> int:
    pass


def add(public_key: PublicKey, ciphertext_a: int, ciphertext_b: int) -> int:
    pass


def multiply(public_key: PublicKey, ciphertext_a: int, plaintext_b: int) -> int:
    pass
