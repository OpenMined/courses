from collections import namedtuple
import secrets
from typing import List, Tuple

import numpy as np  # type: ignore

PrivateKey = namedtuple("PrivateKey", ["lam", "mu"])
PublicKey = namedtuple("PublicKey", ["g", "n", "n_squared"])

DEFAULT_BIT_LENGTH = 3072


def generate_primes(n: int) -> List[int]:
    # https://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n-in-python/3035188#3035188
    """ Input n>=6, Returns an array of primes, 2 <= p < n """
    sieve = np.ones(n // 3 + (n % 6 == 2), dtype=bool)
    for i in range(1, int(n ** 0.5) // 3 + 1):
        if sieve[i]:
            k = 3 * i + 1 | 1
            sieve[k * k // 3 :: 2 * k] = False
            sieve[k * (k - 2 * (i & 1) + 4) // 3 :: 2 * k] = False
    primes = np.r_[2, 3, ((3 * np.nonzero(sieve)[0][1:] + 1) | 1)]
    return [int(n) for n in primes]


def L(n: int, x: int) -> int:
    return (x - 1) // n


def create_key_pair(
    bit_length: int = DEFAULT_BIT_LENGTH,
) -> Tuple[PrivateKey, PublicKey]:
    primes = generate_primes(2 ** (bit_length // 2))

    p = secrets.choice(primes)
    q = secrets.choice(primes)
    n = p * q

    while p == q or n.bit_length() != bit_length or np.gcd(n, (p - 1) * (q - 1)) != 1:
        p = secrets.choice(primes)
        q = secrets.choice(primes)
        n = p * q

    n_squared = n ** 2
    g = secrets.randbelow(n_squared - 1) + 1
    public_key = PublicKey(g, n, n_squared)

    lam = int(np.lcm(p - 1, q - 1))

    try:
        mu = pow(L(n, pow(g, lam, n_squared)), -1, n)
    except ValueError:
        return create_key_pair(bit_length)

    private_key = PrivateKey(lam, mu)
    return private_key, public_key


def encrypt(public_key: PublicKey, plaintext: int) -> int:
    g, n, n_squared = public_key
    r = secrets.randbelow(n)
    return (pow(g, plaintext, n_squared) * pow(r, n, n_squared)) % n_squared


def decrypt(private_key: PrivateKey, public_key: PublicKey, ciphertext: int) -> int:
    lam, mu = private_key
    _, n, n_squared = public_key
    return (L(n, pow(ciphertext, lam, n_squared)) * mu) % n


def add(public_key: PublicKey, ciphertext_a: int, ciphertext_b: int) -> int:
    return (ciphertext_a * ciphertext_b) % public_key.n_squared


def multiply(public_key: PublicKey, ciphertext_a: int, plaintext_b: int) -> int:
    if plaintext_b == 0:
        return encrypt(public_key, 0)

    if plaintext_b == 1:
        encrypted_zero = encrypt(public_key, 0)
        return add(public_key, ciphertext_a, encrypted_zero)

    return pow(ciphertext_a, plaintext_b, public_key.n_squared)
