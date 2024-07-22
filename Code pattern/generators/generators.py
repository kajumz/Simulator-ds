import random


def username_generator(n, first_names=None, last_names=None):
    """generators for random first, last names"""""
    default_first_names = ["John", "Jane", "Michael", "Emily"]
    default_last_names = ["Smith", "Johnson", "Williams", "Jones"]
    if first_names is None:
        first_names = default_first_names
    if last_names is None:
        last_names = default_last_names
    for i in range(1, n+1):
        user = {}
        user['id'] = i
        user['first_name'] = random.choice(first_names)
        user['last_name'] = random.choice(last_names)
        yield user
def data_generator(n):
    """generators for 2 numbers"""
    for i in range(n):
        yield i, random.randint(0, 100)

# Example of use
#custom_first_names = ["Max", "Sophia", "Liam"]
#custom_last_names = ["Miller", "Davis", "Garcia"]
#for user in username_generator(3, custom_first_names, custom_last_names):
#    print(user['id'], user['first_name'], user['last_name'])

# Example of output:
# 1 Max Garcia
# 2 Liam Davis
# 3 Max Miller
