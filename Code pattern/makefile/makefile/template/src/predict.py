import random


def delivery_time():
    return random.randint(1, 10)


if __name__ == "__main__":
    print(f"Predicted delivery time: {delivery_time()} days")
