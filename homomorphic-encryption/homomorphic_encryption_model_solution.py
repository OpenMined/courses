from typing import List, Tuple

from paillier import add, encrypt, multiply, PublicKey


def encrypted_celsius_to_fahrenheit(public_key: PublicKey, ciphertext: int) -> int:
    """
    Returns an encrypted integer representing 1/10ths of a degree Fahrenheit
    °F = °C * 1.8 + 32
    """
    # First multiply the ciphertext by 18.
    multiplied_by_18 = multiply(public_key, ciphertext, 18)
    # Now encrypt 320.
    encrypted_320 = encrypt(public_key, 320)
    # Finally add them together. The result will need to be divided by 10.
    return add(public_key, multiplied_by_18, encrypted_320)


def encrypted_price_calculator(
    public_key: PublicKey,
    # a list of (encrypted price, plaintext quantity) tuples
    encrypted_cart: List[Tuple[int, int]],
) -> int:
    """
    Returns the encrypted sum of multiplying each encrypted price by an unencrypted quantity
    """
    item_subtotals = [
        multiply(public_key, price, quantity) for price, quantity in encrypted_cart
    ]

    total = encrypt(public_key, 0)
    for item_subtotal in item_subtotals:
        total = add(public_key, total, item_subtotal)

    return total
