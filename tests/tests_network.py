import random
from components.network import Helper


def test_trim_data(size=1000):
    lengths = [random.randint(1000, 1050) for i in range(size)]
    lengths.sort()
    data = [[[0 for j in range(random.choice(lengths))], True] for i in range(size)]
    data = Helper.trim_data(data, lengths[0])
    for i in range(1, size):
        assert len(data[i][0]) == len(data[i - 1][0])
