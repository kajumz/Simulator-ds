import metrics


def test_non_int_clicks():
    try:
        metrics.ctr(1.5, 2)
    except TypeError:
        pass
    else:
        raise AssertionError("Non int clicks not handled")


def test_non_int_views():
    try:
        metrics.ctr(2, 3.5)
    except TypeError:
        pass
    else:
        raise AssertionError("Non int views not handled")


def test_non_positive_clicks():
    try:
        metrics.ctr(-1, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("Non positive int clicks not handled")



def test_non_positive_views():
    try:
        metrics.ctr(2, -11)
    except ValueError:
        pass
    else:
        raise AssertionError("Non positive int views not handled")


def test_clicks_greater_than_views():
    try:
        metrics.ctr(100, 50)
    except ValueError:
        pass
    else:
        raise AssertionError("clicks less than views")


def test_zero_views():
    try:
        metrics.ctr(1, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("Non positive int clicks not handled")

