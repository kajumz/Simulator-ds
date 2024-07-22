#solution.py

import os
import sys
from typing import List

import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()
path = os.path.join(sys.path[0], os.environ['data_path'])

class User(BaseModel):
    user_id: int
    time: int
    popular_streamers: List


def process_data(path_from: str, time_now: int = 6147):
    """Function process

    Parameters
    ----------
    path_from : str
        path to read data
    time_now : int
        time to filter data

    Returns
    -------
    data: pandas.DataFrame
        dataframe after proccessing
    """
    column_names = ["uid", "session", "streamer_name", "time_start", "time_end"] # Specify column names
    dat = pd.read_csv(path_from, names=column_names)
    active_streams = dat[(dat['time_start'] < time_now) & (dat['time_end'] > time_now)]
    return active_streams


def recomend_popularity(data: pd.DataFrame):
    """Recomend Popularity

    Parameters
    ----------
    data : pd.DataFrame

    Returns
    -------
    popular_streamers: List
    """
    popularity = data.groupby('streamer_name')['uid'].count().sort_values(ascending=False)

    popular_streamers = list(popularity.index)

    return popular_streamers


@app.get("/popular/user/{user_id}")
async def get_popularity(user_id: int, time: int = 6147):
    """Fast Api Web Application

    Parameters
    ----------
    user_id : int
        user id
    time : int, optional
        time, by default 6147

    Returns
    -------
    user: json
        user informations
    """
    data = process_data(path, time)
    popular_streamers = recomend_popularity(data)
    user = User(user_id=user_id, time=time, popular_streamers=popular_streamers)
    return user


def main() -> None:
    """Run application"""
    uvicorn.run("solution:app", host="localhost")


if __name__ == "__main__":
    main()
