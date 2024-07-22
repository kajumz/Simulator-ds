from typing import Iterable


def get_bound_types(space):
    """
    Get parameter's type
        - 'uniform': uniform distribution [a, b]
        - 'quniform': uniform distribution [a, b] with step q
        - 'quniform_int': uniform distribution [a, b] with integer step q
        - 'loguniform': log-uniform distribution [log10(a), log10(b)]
        - 'choice' : set of options {A, B, C, ...}
        - 'const': any single value

    Parameters
    ----------
    space : dict
        Boundaries

    Returns
    -------
    dict
        Boundaries type

    
    """
    btypes = {}

    for param, bounds in space.items():

        if isinstance(bounds, str):
            btype = "const"

        elif isinstance(bounds, Iterable):
            if isinstance(bounds, set):
                btype = "choice"

            elif isinstance(bounds, tuple):

                if len(bounds) == 2:
                    btype = "uniform"

                elif len(bounds) == 3:

                    if bounds[2] == "log":
                        btype = "loguniform"

                    elif isinstance(bounds[2], int):
                        btype = "quniform_int"

                    elif isinstance(bounds[2], float):
                        btype = "quniform"

                    else:
                        raise ValueError(f"Unknown bounds type: {bounds}")

                else:
                    raise ValueError(f"Unknown bounds type: {bounds}")

            else:
                raise ValueError(f"Unknown bounds type: {bounds}")

        else:
            btype = "const"

        btypes[param] = btype

    return btypes


def fix_params(params, space):
    """
    Normalize parameters value according to defined space:
        - 'quniform': round param value with defined step
        - 'constant': replace parameter's value with defined constant

    Parameters
    ----------
    params : dict
        Parameters
    space : dict
        Boundaries

    Returns
    -------
    dict
        Normalized parameters

    
    """
    params = dict(params)
    btypes = get_bound_types(space)

    for param, bounds in space.items():
        if btypes[param] in ["quniform", "quniform_int"]:
            a, b, q = bounds
            params[param] = qround(params[param], a, b, q)
            # if space is grid, then find the closest rounding
        if btypes[param] in ["uniform", "loguniform"]:
            a, b = bounds[:2]
            params[param] = min(params[param], b)
            params[param] = max(a, params[param])
            # if space is uniform, check if value is in borders
        elif btypes[param] == "choice":
            x = params[param]
            if isinstance(x, int) or isinstance(x, float):
                z = list(bounds)[0]
                for y in bounds:
                    if abs(x - y) < abs(x - z):
                        z = y
                params[param] = z
            # if space is closed set of values, find the closest
            elif isinstance(list(bounds)[0], str) and (x not in bounds):
                params[param] = list(bounds)[0]
            # if space is closed set of strings, check if in set
        elif btypes[param] == "const":
            params[param] = bounds
            # if space is exact value, then this is a value

    return params


def ranking(ser):
    """
    Make rank transformation.

    Parameters
    ----------
    ser : Series of float
        Values for ranking. None interpreted as worst.

    Returns
    -------
    Series of int
        Ranks (1: highest, N: lowest)

    
    """
    ser = ser.fillna(ser.min())

    rnk = ser.rank(method="dense", ascending=False)
    rnk = rnk.astype(int)

    return rnk


def qround(x, a, b, q):
    """
    Convert x to one of [a, a+q, a+2q, .., b]

    Parameters
    ----------
    x : int or float
        Input value. x must be in [a, b].
        If x < a, x set to a.
        If x > b, x set to b.
    a : int or float
        Boundaries. b must be greater than a. Otherwize b set to a.
    b : int or float
        Boundaries. b must be greater than a. Otherwize b set to a.
    q : int or float
        Step value. If q and a are both integer, x set to integer too.

    Returns
    -------
    int or float
        Rounded value

    
    """
    # Check if a <= x <= b
    b = max(a, b)
    x = min(max(x, a), b)

    # Round x (with defined step q)
    x = a + ((x - a) // q) * q

    # Convert x to integer
    if isinstance(a + q, int):
        x = int(x)

    return x


def main():
    """Use-case demo"""
    space = {
        # quantative space for depth from a=2 to b=8 with step=2:
        # [a, a + step, a + 2*step, ..., b] = [2, 3, 4, ..., 10]
        "max_depth": (2, 10, 1),
        # exact set of allowed values for # of leaves:
        "num_leaves": {1, 2, 4, 8, 16, 32, 64, 128},
        # also works with strings
        "tree_type": {"Depth-wise", "Leaf-wise", "Symmetric"},
        # uniform values from 0.5 to 0.8
        "bagging_fraction": (0.5, 0.8),
        # log-uniform values from 0.1 to 0.9
        "feature_fraction": (0.2, 0.7, "log"),
        # constant (you can skip it in params)
        "learning_rate": 0.0042,
    }

    params = {
        "max_depth": 5,
        "num_leaves": 30,
        "tree_type": "Symmetric",
        "bagging_fraction": 0.666,
        "feature_fraction": 0.777,
    }

    print("Initial parameters:")
    print(params, end="\n\n")

    params = fix_params(params, space)

    print("Prameters after ajustment:")
    print(params, end="\n\n")


if __name__ == "__main__":
    main()
