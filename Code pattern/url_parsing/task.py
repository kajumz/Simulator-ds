import re
from collections import Counter
from typing import Dict

import requests


def parse_urls(message: str) -> Dict[str, int]:
    urls = re.findall(r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)"
                      r"(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|"
                      r"[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))", message)

    # Нормализуем URL-адреса, удаляя "www", "http://" или "https://"
    normalized_urls = [re.sub(r"(?i)(?:https?://)?(?:www\.)?", "", url) for url in urls]

    # Используем Counter для подсчета количества каждого URL-адреса
    url_counts = Counter(normalized_urls)

    # Проверяем доступность каждого URL-адреса
    #for url in url_counts.keys():
    #    try:
    #        response = requests.head("http://" + url)
    #        if response.status_code != 200:
    #            del url_counts[url]
    #    except requests.exceptions.RequestException:
    #        del url_counts[url]

    return dict(url_counts)


if __name__ == "__main__":
    message = (
        "Check out this link www.example.com, example.com and"
        " also https://www.xn--80ak6aa92e.com/"
        " also www.xn--80ak6aa92e.com"
        " also xn--80ak6aa92e.com"
        " also apple.com"
        " Don't miss this great opportunity!"
        " www.google.com."
        " hello.ru"
    )
    print(parse_urls(message))
