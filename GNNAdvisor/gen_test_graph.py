from math import floor
from random import random


def rand(): return floor(random()*100)


set = []
for _ in range(400):
    a = rand()
    b = rand()
    if a != b and [a, b] not in set:
        print('{} {}\n{} {}'.format(a, b, b, a))
        set.append([a, b])
        set.append([b, a])
