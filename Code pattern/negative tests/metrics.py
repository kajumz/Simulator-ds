def ctr(clicks: int, views: int) -> float:
    """Click-through Rate."""

    # Check that the values are integers
    if not isinstance(clicks, int):
        raise TypeError("clicks must be an integer")

    if not isinstance(views, int):
        raise TypeError("views must be an integer")

    # Check that the values are positive
    if clicks < 0:
        raise ValueError("clicks must be positive")

    if views < 0:
        raise ValueError("views must be positive")

    # Check if clicks are greater than views
    if views < clicks:
        raise ValueError("clicks must be less than or equal to views")

    # Calculate the clickthrough rate
    if views:
        return clicks / views
    else:
        raise ZeroDivisionError("views must be greater than zero")
