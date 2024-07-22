from functools import reduce
from typing import List



def sales_with_tax(sales: List[float], tax_rate: float, threshold: float = 300) -> List[float]:
    """Из списка продаж фильтровать те, что превышают определенную сумму, и применить к ним налог"""
    li = list(filter(lambda x: x > threshold, sales))
    res = list(map(lambda x: x*(1+tax_rate), li))
    return res



def sum_sales(sales: List[float], threshold: float = 300) -> float:
    """ Суммировать продажи после фильтрации по минимальной сумме продажи"""
    filtered_sales = list(filter(lambda x: x > threshold, sales))
    total_sales = reduce(lambda x, y: float(x + y), filtered_sales)
    return total_sales


def average_age(ages: List[int], threshold: int = 30) -> float:
    """Найти средний возраст клиентов, чей возраст превышает определенный порог."""
    filter_age = list(filter(lambda x: x > threshold, ages))
    total = reduce(lambda x, y: x+y, filter_age)
    return float(total / len(filter_age))


def increased_prices(prices: List[float], increase_rate: int = 0.2, threshold: float = 300) -> List[float]:
    """Увеличить цену каждого товара на 20% и отфильтровать те, чья итоговая цена превышает определенный порог"""
    up_price = map(lambda x: x*(1+increase_rate), prices)
    li = list(filter(lambda x: x > threshold, up_price))
    return li

def weighted_sale_price(sales: List[float]) -> float:
    """Рассчитайте средневзвешенную цену проданных товаров. Взвешивать нужно на количество продаж. Т.е. количество выступает как вес в формуле средневзвешенного"""
    #weigh = map(lambda x: float(x[0] / x[1]), sales)
    #return reduce(lambda x, y: x+y, weigh)
