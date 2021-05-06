from homomorphic_encryption_model import (
    encrypted_celsius_to_fahrenheit,
    encrypted_price_calculator,
)
from paillier import create_key_pair, decrypt, encrypt

# Use a short bit length for testing, otherwise this may take a long time
TEST_BIT_LENGTH = 32


def test_encrypted_celsius_to_fahrenheit():
    private_key, public_key = create_key_pair(bit_length=TEST_BIT_LENGTH)

    temperature_in_celsius = 23
    temperature_in_fahrenheit = 73.4

    encrypted_input = encrypt(public_key, temperature_in_celsius)
    encrypted_output = encrypted_celsius_to_fahrenheit(public_key, encrypted_input)
    decrypted_output = decrypt(private_key, public_key, encrypted_output)
    # Note that the Paillier cryptosystem only deals with integers and cannot handle
    # encrypted division because it is only a partial homomorphic encryption scheme
    scaled_output = decrypted_output / 10

    assert scaled_output == temperature_in_fahrenheit


def test_encrypted_price_calculator():
    private_key, public_key = create_key_pair(bit_length=TEST_BIT_LENGTH)

    cart = [
        # (price, quantity)
        (2000, 1),
        (120, 5),
        (1999, 3),
    ]
    expected_price = 8597

    # Note that the items are somewhat anonymized by the removal of the highly
    # identifying price information, but the plaintext quantity could still provide
    # information which could deanonymize the item data
    encrypted_cart = [
        (encrypt(public_key, price), quantity) for price, quantity in cart
    ]
    encrypted_price = encrypted_price_calculator(public_key, encrypted_cart)
    decrypted_price = decrypt(private_key, public_key, encrypted_price)

    assert decrypted_price == expected_price
