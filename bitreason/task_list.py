### Utils


def to_int(bits: list[int]) -> int:
    return int("".join(str(b) for b in bits), 2)


def to_list(num: int) -> list[int]:
    return [int(b) for b in bin(num)[2:]]


def zero_pad(bits: list[int], left: bool = True, size: int = 16) -> list[int]:
    if len(bits) >= size:
        return bits[:size]
    pad_len = size - len(bits)
    padding = [0] * pad_len
    return padding + bits if left else bits + padding


### Tasks
# Names of functions are extracted and used for reporting/visualization, so make them descriptive


def not_(bits):
    # bitwise NOT
    return [1 - b for b in bits]


def and_(bits):
    # bitwise AND between halves, zero pad on left
    mid = len(bits) // 2
    return zero_pad([bits[i] & bits[i + mid] for i in range(mid)])


def or_(bits):
    # bitwise OR between halves, zero pad on left
    mid = len(bits) // 2
    return zero_pad([bits[i] | bits[i + mid] for i in range(mid)])


def xor(bits):
    # bitwise XOR between halves, zero pad on left
    mid = len(bits) // 2
    return zero_pad([bits[i] ^ bits[i + mid] for i in range(mid)])


def sum_(bits):
    # add halves as binary numbers, zero pad on left
    mid = len(bits) // 2
    a = to_int(bits[:mid])
    b = to_int(bits[mid:])

    res = a + b
    return zero_pad(to_list(res))


def shift(bits):
    # Shift left by 8
    shift = 8
    return bits[shift:] + bits[:shift]


def flip(bits):
    # Flip
    return bits[::-1]


def tile(bits):
    # Repeat first 4 bits across list, zero pad on left
    seg = 4
    return zero_pad(bits[:seg] * (len(bits) // seg))


def count(bits):
    # count number of 1s and return as binary number, zero pad on left
    num_ones = sum(bits)
    bin_str = bin(num_ones)[2:]
    return [int(b) for b in bin_str] + [0] * (len(bits) - len(bin_str))


def separate(bits):
    # move all 1s to front, all 0s to back
    num_ones = sum(bits)
    num_zeros = len(bits) - num_ones
    return [1] * num_ones + [0] * num_zeros


def divisors(bits):
    # for each i in 2 to len(bits)+1, output 1 if i divides the number represented by bits, else 0
    num = to_int(bits)
    divs = []
    for i in range(2, len(bits) + 2):
        if num % i == 0:
            divs.append(1)
        else:
            divs.append(0)
    return divs


def gcd(bits):
    # gcd of halves as uints, zero pad on left
    mid = len(bits) // 2
    a = to_int(bits[:mid])
    b = to_int(bits[mid:])
    while b:
        a, b = b, a % b
    return zero_pad(to_list(a))


task_list = [
    not_,
    and_,
    or_,
    xor,
    sum_,
    shift,
    flip,
    tile,
    count,
    separate,
    divisors,
    gcd,
]
