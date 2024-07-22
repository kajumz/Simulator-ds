import random


def data_generator(n):
    # YOUR CODE
    for i in range(n):
        a = []
        a.insert(0, i)
        a.insert(1, random.randint(0, 100))
        yield a
# Example of use
for data in data_generator(3):
    print(data)

# Example of output:
# 0 49
# 1 27
# 2 88
