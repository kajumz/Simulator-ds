from typing import Callable


def memoize(func: Callable) -> Callable:
    """Memoize function"""
    cache = {}
    def memoized(*args, **kwargs):
        key = (str(args), frozenset(kwargs.items()))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return memoized


