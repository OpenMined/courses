from paillier import (
    add,
    create_key_pair,
    decrypt,
    encrypt,
    multiply,
)

# Use a short bit length for testing, otherwise this may take a long time
# Note a very short bit length may result in collisions and test failures
TEST_BIT_LENGTH = 32


def test_encrypt_and_decrypt():
    private_key, public_key = create_key_pair(bit_length=TEST_BIT_LENGTH)
    plaintext = 123

    ciphertext = encrypt(public_key, plaintext)
    assert ciphertext != plaintext

    decrypted = decrypt(private_key, public_key, ciphertext)
    assert decrypted == plaintext


def test_add():
    private_key, public_key = create_key_pair(bit_length=TEST_BIT_LENGTH)
    a = 123
    b = 37
    expected = (123 + 37) % public_key.n

    ciphertext_a = encrypt(public_key, a)
    ciphertext_b = encrypt(public_key, b)

    encrypted_result = add(public_key, ciphertext_a, ciphertext_b)
    result = decrypt(private_key, public_key, encrypted_result)

    assert result == expected


def test_multiply():
    private_key, public_key = create_key_pair(bit_length=TEST_BIT_LENGTH)
    a = 123
    b = 25
    expected = (123 * 25) % public_key.n

    ciphertext_a = encrypt(public_key, a)
    encrypted_result = multiply(public_key, ciphertext_a, b)
    result = decrypt(private_key, public_key, encrypted_result)

    assert result == expected


def test_multiply_by_zero():
    private_key, public_key = create_key_pair(bit_length=TEST_BIT_LENGTH)
    a = 123
    b = 0
    expected = 0
    naive_encrypted_result = 1
    ciphertext_a = encrypt(public_key, a)

    encrypted_result = multiply(public_key, ciphertext_a, b)
    assert encrypted_result != naive_encrypted_result

    result = decrypt(private_key, public_key, encrypted_result)
    assert result == expected


def test_multiply_by_one():
    private_key, public_key = create_key_pair(bit_length=TEST_BIT_LENGTH)
    a = 123
    b = 1
    expected = 123
    ciphertext_a = encrypt(public_key, a)
    naive_encrypted_result = ciphertext_a

    encrypted_result = multiply(public_key, ciphertext_a, b)
    assert encrypted_result != naive_encrypted_result

    result = decrypt(private_key, public_key, encrypted_result)
    assert result == expected
