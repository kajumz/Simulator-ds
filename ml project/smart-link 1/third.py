import random
from collections import defaultdict
import uvicorn
from fastapi import FastAPI



app = FastAPI()
pending_clicks = {} # связка клика с оффером key: click value: offer_id
offer_rewards = defaultdict(float) # связка награды с оффером key: offer_id value: reward
offer_clicks = defaultdict(int) # связь оффера с кликом, key: offer_id value: количество кликов
offer_actions = defaultdict(int) # для действий для количества действий key: offer_id value:_количество конверсий

def reset_statistics():
    """function to reset statistics"""
    global offer_clicks, offer_rewards, offer_actions
    offer_clicks = defaultdict(int)
    offer_rewards = defaultdict(float)
    offer_actions = defaultdict(int)

@app.on_event("startup")
async def startup_event():
    print("Application starting, resetting statistics...")
    reset_statistics()

@app.put("/feedback/")
def feedback(click_id: int, reward: float) -> dict:
    """/feedback отдаёт информацию о том, что случилось с кликом, сколько денег (в центах) мы за него получили.
    Если конверсии не было, то reward=0 (напомним, что конверсия — это когда пользователь перешёл на оффер и выполнил нужное действие: скачал приложение, закинул деньги в казино и т.д.).
    В ответе на этот запрос ожидается сообщение, какая информация была усвоена:
    Какой click ID получен
    Какой offer ID обновлён (обратите внимание, что в этот запрос оффер уже не передаётся)
    Была ли конверсия: is_conversion
    И сколько денег мы получили за этот клик
    """
    conv = reward > 0
    offer_id = pending_clicks[click_id]
    offer_actions[offer_id] += int(conv)
    offer_rewards[offer_id] += reward
    del pending_clicks[click_id]
    response = {
        "click_id": click_id,
        "offer_id": offer_id,
        "is_conversion": reward > 0,
        "reward": reward
    }
    return response


@app.get("/offer_ids/{offer_id}/stats/")
def stats(offer_id: int) -> dict:
    """/stats с актуальной информацией по выбранному офферу о том,
        какие были совершены клики, какие получены конверсии,
        какое среднее число денег мы получаем за клик и т.д.

        offer_id – оффер, для которого возвращаем статистику
        clicks – количество кликов или количество показов оффера. Потому что в нашем случае на каждый клик, пришедший в /sample, мы рекомендуем какой-то оффер, который показывается пользователю.
        conversions – количество конверсий, т.е. совершенных целевых действий по этому офферу
        reward – суммарная награда, полученная за все конверсии по этому офферу
        CR (conversion rate) – отношение конверсий к кликам
        RPC (revenue per click) – средняя выручка на клик
        """
    clicks = offer_clicks.get(offer_id, 0)
    conversions = offer_actions.get(offer_id, 0)
    reward = float(offer_rewards.get(offer_id, 0))

    cr = conversions / clicks if clicks > 0 else 0
    rpc = reward / clicks if clicks > 0 else 0


    response = {
        "offer_id": offer_id,
        "clicks": clicks,
        "conversions": conversions,
        "reward": reward,
        "cr": cr,
        "rpc": rpc,
    }
    return response


@app.get("/sample/")
def sample(click_id: int, offer_ids: str) -> dict:
    """На первые 100 кликов, оправленные в сервис, выбирался случайный (для инициализации)
    На последующие выбирался тот (среди баннеров-кандидатов),
    который максимизирует RPC. Выбирать необходимо среди баннеров-кандидатов,
    переданных в аргументе offer_ids
    Если не нашли подходящего оффера, возвращаем самый первый.
    Например, на вход пришли офферы offers_ids: [45, 67].
    Но статистика по обоим офферам нулевая. Тогда выбираем 45
    Кроме "click_id" в ответ добавляется поле "sampler", которое содержит значение "random" или "greedy",
    в зависимости от того, на основе чего был выбран оффер для клика."""
    # Parse offer IDs
    offers_ids = [int(offer) for offer in offer_ids.split(",")]

    epsilon = 0.1

    if random.random() < epsilon:
        offer_id = random.choice(offers_ids)
        sampler = 'random'
    else:
        max_rpc_offer = max(offers_ids, key=lambda offer: offer_rewards.get(offer, 0) / max(offer_clicks.get(offer, 0), 1))
        offer_id = max_rpc_offer if offer_actions.get(max_rpc_offer, 0) > 0 else offers_ids[0]
        sampler = 'greedy'
    offer_clicks[offer_id] += 1 #offer_clicks.get(offer_id, 0) + 1
    pending_clicks[click_id] = offer_id

    # Prepare response
    response = {
        "click_id": click_id,
        "offer_id": offer_id,
        "sampler": sampler,
    }

    return response


def main() -> None:
    """Run application"""
    uvicorn.run("second:app", host="localhost")


if __name__ == "__main__":
    main()
