import math
def movement_microseconds(length_m):
    value = length_m / 5 * 1000000
    whole = math.floor(value)
    return whole + int(value - whole >= 0.5)
