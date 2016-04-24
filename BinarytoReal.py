from __future__ import division


def precision(high_limit, low_limit, number_bits):
    return float(high_limit - low_limit)/float(((2 ** number_bits) - 1))


def convert(high_limit, low_limit, number_bits, binary_number):
    return low_limit + (precision(high_limit, low_limit, number_bits) * to_int(binary_number))


def to_int(binary_number):
    return int(binary_number, 2)
