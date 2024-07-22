import metrics


def test_profit() -> None:
    """unit-test for profit"""
    assert metrics.profit([1, 2, 3], [1, 1, 1]) == 3


def test_margin() -> None:
    """unit-test for margin"""
    assert metrics.margin([1, 2, 3], [1, 1, 1]) == 0.5


def test_markup() -> None:
    """unit-test for markup"""
    assert metrics.markup([1, 2, 3], [1, 1, 1]) == 1
