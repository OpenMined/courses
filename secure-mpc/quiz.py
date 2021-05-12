import pydagogy as pgy
import numpy as np
from typing import List

q0 = pgy.tests.ValueTest(25)
q0.hint = "Modulo (%) is the remainder of a division \n\n s = ( s\u2081 + s\u2082 ) % Q"
q0.solution = "(10 + 74 ) % 59 = " + str(25)
q0.success = "Correct!"


q1 = pgy.tests.ValueTest(57)
q1.hint = "Plug the variables mentioned above into s\u2082 = Q - (s\u2081 % Q)"
q1.solution = "s\u2082 = " + str(57)
q1.success = "Nicely done!"
# q1 explanation:
# final_share = 59 - ( 9 % 59) + 7
# also
# final_share = 57

q2 = pgy.tests.ValueTest(686)
q2.hint = """Iterate from i to n-1 and append shares.
            \n Also, sum all the shares before the modulo operation
            \n Plus, don't forget tuples are more secure"""
q2.solution = """\n
                # make an iterable
                share_lst = list()\n

                # generate shares randomly except for the final share
                for i in range(n - 1):\n

                    share_lst.append(randint(0,r))\n\n

                final_share = r - (sum(share_lst) % r) + s\n

                share_lst.append(final_share)\n

                # return a tuple of shares
                return tuple(share_lst)"""
q2.success = "Nicely done!"

# q2 solution
def n_share(s, r, n):
    """
    s = secret
    r = randomness
    n = number of nodes, workers or participants
    """
    share_lst = list()

    for i in range(n - 1):
        share_lst.append(randint(0, r))

    final_share = r - (sum(share_lst) % r) + s

    share_lst.append(final_share)

    return tuple(share_lst)


q3 = pgy.tests.ValueTest(-93)
q3.hint = "Reconstruct x - a and y - b by adding the Alice and Bob's [x-a] and [y-b]"
q3.solution = "z_alice = " + str(-93)
q3.success = "Excellent!"

q4 = pgy.tests.ValueTest(117)
q4.hint = "Remember Bob adds (x-a)(x-b) to his share. z_bob = c_bob + (a_bob)(y-b) + (b_bob)(x-a) + (x-a)(y-b) "
q4.solution = "z_bob = " + str(117)
q4.success = "Correct! Bob's share is -15 + (6)(22) = 117"


## Part 4


class EncodeTest(pgy.tests.BaseTest):
    def check(self, test_function):
        def correct_function(x: float, base: int, precision: int) -> int:
            return round(x * base ** precision)

        message = "The encode function should return an integer!"

        value = test_function(1.2, 10, 3)
        if not pgy.asserts.assert_true(isinstance(value, int), message):
            return

        for x, base, precision in [
            (1.2, 10, 3),
            (-0.002, 10, 3),
            (1.0008, 10, 3),
            (1.2, 2, 5),
            (-0.02, 2, 5),
            (1.02, 2, 5),
        ]:
            expected = correct_function(x, base, precision)
            test_output = test_function(x, base, precision)
            message = f"encode({x}, {base}, {precision}) should output {expected}, not {test_output}"
            if not pgy.asserts.assert_true(expected == test_output, message):
                return

        self.success


# Create the test and add feedback for students
test_encode = EncodeTest()
test_encode.hint = (
    "You can use the round() function. Don't forget to use the base argument."
)
test_encode.solution = """
def encode(x: float, base: int, precision: int) -> int:
    return round(x * base ** precision)
"""
test_encode.success = "Nicely done! 🙌"


class DecodeTest(pgy.tests.BaseTest):
    def check(self, test_function):
        def correct_function(x: float, base: int, precision: int) -> float:
            return x / base ** precision

        for x, base, precision in [
            (1200, 10, 3),
            (-2, 10, 3),
            (1001, 10, 3),
            (38, 2, 5),
            (-1, 2, 5),
            (33, 2, 5),
        ]:
            expected = correct_function(x, base, precision)
            test_output = test_function(x, base, precision)
            message = f"decode({x}, {base}, {precision}) should output {expected}, not {test_output}"
            if not pgy.asserts.assert_true(expected == test_output, message):
                return

        self.success


# Create the test and add feedback for students
test_decode = DecodeTest()
test_decode.hint = "Don't forget to use the base."
test_decode.solution = """
def decode(x: float, base: int, precision: int) -> float:
    return x / base ** precision
"""
test_decode.success = "Well done!"


class FPTruncateTest(pgy.tests.BaseTest):
    def check(self, test_function):
        def correct_function(x: int, base: int, precision: int) -> int:
            return x // base ** precision

        for x, base, precision in [
            (1200, 10, 3),
            (-2, 10, 3),
            (1001, 10, 3),
            (38, 2, 5),
            (-1, 2, 5),
            (33, 2, 5),
        ]:
            expected = correct_function(x, base, precision)
            test_output = test_function(x, base, precision)
            message = f"truncate({x}, {base}, {precision}) should output {expected}, not {test_output}"
            if not pgy.asserts.assert_true(expected == test_output, message):
                return

        self.success


# Create the test and add feedback for students
test_fp_truncate = FPTruncateTest()
test_fp_truncate.hint = "You can use // to perform true division"
test_fp_truncate.solution = """
def truncate(x: int, base: int, precision: int) -> int:
    return x // base ** precision
"""
test_fp_truncate.success = "All good!"


class FPMulTest(pgy.tests.BaseTest):
    def check(self, test_function):
        def encode(x: float, base: int, precision: int) -> int:
            return round(x * base ** precision)

        def truncate(x: int, base: int, precision: int) -> int:
            return x // base ** precision

        def mul(x: int, y: int, base: int, precision: int) -> int:
            z = x * y
            z = truncate(z, base, precision)
            return z

        base = 10
        precision = 3
        x_fp = encode(1.2, base, precision)
        y_fp = encode(2.0, base, precision)
        expected = mul(x_fp, y_fp, base, precision)
        test_output = test_function(x_fp, y_fp, base, precision)
        message = f"mul({x_fp}, {y_fp}, {base}, {precision}) should output {expected}, not {test_output}"
        if not pgy.asserts.assert_true(expected == test_output, message):
            return

        self.success


# Create the test and add feedback for students
test_fp_mul = FPMulTest()
test_fp_mul.hint = "Don't forget to use truncation!"
test_fp_mul.solution = """
def mul(x: int, y: int, base: int, precision: int) -> int:
    z = x * y
    z = truncate(z, base, precision)
    return z
"""
test_fp_mul.success = "Nicely done!"


base = 10
precision = 3
bits = 32  # On how many bits we encode our fixed precision values and shares


def encode(x: float) -> int:
    return round(x * base ** precision)


def decode(x: int) -> int:
    return x // base ** precision


def secret_share(x: int) -> List[int]:
    share_1 = np.random.randint(0, 2 ** bits)
    share_2 = (x - share_1) % 2 ** bits

    shifted_shares = [share_1 - 2 ** (bits - 1), share_2 - 2 ** (bits - 1)]
    return shifted_shares


def decrypt(shares: List[int]) -> int:
    sum_shares = sum(shares) % 2 ** bits
    if sum_shares > 2 ** (bits - 1) - 1:
        sum_shares -= 2 ** bits
    return sum_shares


def spdz_mul(x_shares: List[int], y_shares: List[int]) -> List[int]:
    z = decrypt(x_shares) * decrypt(y_shares)
    z_shares = secret_share(z)
    return z_shares


def wrap_around(shares: List[int]) -> List[int]:
    sum_shares = sum(shares)
    wrap = 0
    while sum_shares >= 2 ** (bits - 1):
        wrap += 1
        sum_shares -= 2 ** bits
    while sum_shares < -(2 ** (bits - 1)):
        wrap -= 1
        sum_shares += 2 ** bits
    return secret_share(wrap)


# checks
x = 4
for i in range(20):
    a, b = secret_share(x)
    assert a >= -(2 ** (bits - 1)) and a < 2 ** (bits - 1)
    assert b >= -(2 ** (bits - 1)) and b < 2 ** (bits - 1)
    y = decrypt([a, b])
    assert x == y


a = 1.2
b = 2.0
for i in range(20):
    a_fp, b_fp = encode(a), encode(b)
    a_shares, b_shares = secret_share(a_fp), secret_share(b_fp)
    c_shares = spdz_mul(a_shares, b_shares)
    c = decrypt(c_shares)


class TruncateTest(pgy.tests.BaseTest):
    def check(self, test_function):
        def correct_function(x_shares: List[int]) -> List[int]:
            theta_shares = wrap_around(x_shares)

            truncated_shares = []
            for x_share, theta_share in zip(x_shares, theta_shares):
                truncated_shares.append(
                    int(x_share / base ** precision)
                    - theta_share * int(2 ** bits / base ** precision)
                )

            return truncated_shares

        for x in [
            12.1,
            829827.2,
            829827.2,
            829827.2,
            829827.2,
            829827.2,
        ]:
            x_fp = encode(x)
            x_shares = secret_share(x_fp)
            x_shares_trunc = correct_function(x_shares)
            x_fp_trunc = decrypt(x_shares_trunc)
            test_output = decrypt(test_function(x_shares))
            message = f"decrypt(truncate(secret_share({x_fp}))) should output {x_fp_trunc}, not {test_output}"
            if not pgy.asserts.assert_true(x_fp_trunc == test_output, message):
                return

        self.success


# Create the test and add feedback for students
test_truncate = TruncateTest()
test_truncate.hint = """
Try to analyze the formula and the explanation provided for the exact truncation.
You will need to call wrap_around().
"""
test_truncate.solution = """
def truncate(x_shares: List[int]) -> List[int]:
    theta_shares = wrap_around(x_shares)
    
    truncated_shares = []
    for x_share, theta_share in zip(x_shares, theta_shares):
        truncated_shares.append(
            int(x_share / base ** precision) - theta_share * int(2 ** bits / base ** precision)
        )
        
    return truncated_shares
"""
test_truncate.success = "Excellent! This was not an easy one 🙌"


class MulTest(pgy.tests.BaseTest):
    def check(self, test_function):
        def truncate(x_shares: List[int]) -> List[int]:
            # Write your code here
            theta_shares = wrap_around(x_shares)

            truncated_shares = []
            for x_share, theta_share in zip(x_shares, theta_shares):
                truncated_shares.append(
                    int(x_share / base ** precision)
                    - theta_share * int(2 ** bits / base ** precision)
                )

            return truncated_shares

        def mul(x_shares: List[int], y_shares: List[int]) -> List[int]:
            # Write your code here
            z_shares = spdz_mul(x_shares, y_shares)
            z_shares = truncate(z_shares)
            return z_shares

        x = 17.2
        y = 27.0
        x_fp, y_fp = encode(x), encode(y)
        x_shares, y_shares = secret_share(x_fp), secret_share(y_fp)
        z_shares = mul(x_shares, y_shares)
        z_fp = decrypt(z_shares)
        expected = decode(z_fp)

        test_output = decode(decrypt(test_function(x_shares, y_shares)))

        message = f"""
decode(decrypt(mul(x_shares, y_shares))) should output {expected}, not {test_output}.
where x_shares = secret_share(encode({x}))
      y_shares = secret_share(encode({y}))
        """
        if not pgy.asserts.assert_true(expected == test_output, message):
            return

        self.success


# Create the test and add feedback for students
test_mul = MulTest()
test_mul.hint = "This is very close to what you did when there was no secret sharing."
test_mul.solution = """
def mul(x_shares: List[int], y_shares: List[int]) -> List[int]:
    z_shares = spdz_mul(x_shares, y_shares)
    z_shares = truncate(z_shares)
    return z_shares
"""
test_mul.success = "Well done! 🙌"


## Part 6

def random_mask() -> int:
    """Return a non-zero random value"""
    value = np.random.randint(- 2 ** (bits - 1), 2 ** (bits - 1))
    if value == 0:
        return random_mask()
    return value


def sub(x: List[int], y: List[int]) -> List[int]:
    """Emulates x - y for shared values"""

    n_party = len(x)
    z = [
        x[party] - y[party]
        for party in range(n_party)
    ]

    return z

mul = spdz_mul


class EqualityTest(pgy.tests.BaseTest):
    def check(self, test_function):
        for x, y in [
            (3, 3),
            (4, 8),
            (-2, -2),
            (-4, 6)
        ]:
            z = test_function(x, y)
            if x == y:
                message = f"equality({x}, {y}) should output 0, not {z}"
                if not pgy.asserts.assert_true(z == 0, message):
                    return
            else:
                message = f"equality({x}, {y}) should not output 0/"
                if not pgy.asserts.assert_true(z != 0, message):
                    return

        self.success


# Create the test and add feedback for students
test_equality = EqualityTest()
test_equality.hint = """
Follow the 4 steps of the protocol. Don't bother about separating the code of the
different parties. You can view the 2 first steps like a preparation, while the 2
next steps really are the private equality test.
"""
test_equality.solution = """
# This is a suggestion, you might have a different and correct implementation

def equality(x: int, y: int) -> int:
    x_sh = secret_share(x)
    y_sh = secret_share(y)
    
    r1 = random_mask()
    r1_sh = secret_share(r1)
    r2 = random_mask()
    r2_sh = secret_share(r2)
    
    aux1_sh = sub(x_sh, y_sh)
    aux2_sh = mul(aux1_sh, r1_sh)
    z_sh = mul(aux2_sh, r2_sh)
    
    z = decrypt(z_sh)
    return z 
"""
test_equality.success = "Excellent! 🙌"


test_rounds = pgy.tests.ValueTest(2)
test_rounds.hint = "Does - require some communication? Does * require some communication?"
test_rounds.solution = "Only the 2 multiplication need a round each, so the total is 2"
test_rounds.success = "Correct!"


test_comp_rounds = pgy.tests.ValueTest(2)
test_comp_rounds.hint = "All the checks X_i == Y_i can be done in parallel!"
test_comp_rounds.solution = "It's 2, since all the check can be done in parallel"
test_comp_rounds.success = "Correct!"