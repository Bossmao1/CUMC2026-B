"""Local timing convention; official response timestamps remain authoritative."""
import math


def movement_microseconds(length_m):
    # Half microseconds round away from zero (Go math.Round convention).
    # The movement rounding tie branch still needs independent EXE confirmation.
    value = length_m / 5 * 1_000_000
    whole = math.floor(value)
    return whole + int(value - whole >= .5)
