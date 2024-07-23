import random
import time

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

app = FastAPI()

@app.post("/parse_url/")
def parse_url(url: str) -> str:
    try:
        with httpx.Client() as client:
            r = client.get(url)
            r.raise_for_status()
            parse_time = 0.1 * random.randint(5, 10) if random.random() < 0.1 else 0.1
            time.sleep(parse_time)

            return f"Parsed {url}"
    except Exception as e:
        return f"Error fetching {url}: {e}"


def run_test(n_requests: int) -> float:
    url = "https://httpbin.org/"
    with TestClient(app) as client:
        ts = time.time()
        for _ in range(n_requests):
            _ = client.post("/parse_url/", params={"url": url})
        return time.time() - ts



if __name__ == "main":
    t = run_test(n_requests=100)
    print(f"Time taken: {t} seconds")