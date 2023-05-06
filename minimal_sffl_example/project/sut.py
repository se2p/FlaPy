import random


def gen_int():
    r_int = random.randint(0, 100)
    if r_int >= 50:
        return "Big number"
    else:
        return "Small number"
