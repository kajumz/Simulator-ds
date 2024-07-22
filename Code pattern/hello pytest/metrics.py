from typing import List


def profit(revenue: List[float], costs: List[float]) -> float:
    return sum(revenue) - sum(costs)


def margin(revenue: List[float], costs: List[float]) -> float:
    return (sum(revenue) - sum(costs)) / sum(revenue)


def markup(revenue: List[float], costs: List[float]) -> float:
    return (sum(revenue) - sum(costs)) / sum(costs)
