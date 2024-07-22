import os
import pickle
import sys
from typing import List

import implicit
import numpy as np
import pandas as pd
#import scipy.sparse
import uvicorn

from fastapi import FastAPI
from pydantic import BaseModel
from scipy.sparse import csr_matrix

app = FastAPI()
path = os.path.join(sys.path[0], os.environ['D:\pythonProject4\middle\recsys\data_recsys.csv'])
model_path = os.path.join(sys.path[0], os.environ["D:\pythonProject4\middle\recsys"])
class User(BaseModel):
    """Class of json output"""
    user_id: int
    personal: List


def process_data(path_from: str):
    """Function process

    Parameters
    ----------
    path_from : str
        path to read data

    Returns
    -------
    data: pandas.DataFrame
        dataframe after proccessing
    sparse_item_user: scipy.sparse.csr_matrix
        sparce item user csr matrix
    """
    column_names = ["uid", "session", "streamer_name", "time_start", "time_end"]  # Specify column names
    data = pd.read_csv(path_from, names=column_names)
    data["all_time"] = data['time_end'] - data['time_start']
    data["uid"] = data["uid"].astype("category")
    data["streamer_name"] = data["streamer_name"].astype("category")
    data["user_id"] = data["uid"].cat.codes
    data["streamer_id"] = data["streamer_name"].cat.codes
    data = data[['user_id', 'streamer_id', 'all_time']]
    sparse_item_user = csr_matrix(
        (data['all_time'], (data['streamer_id'], data['user_id']))
    )

    return data, sparse_item_user


def fit_model(
    sparse_item_user,
    model_path: str,
    iterations: int = 12,
    factors: int = 500,
    regularization: float = 0.2,
    alpha: float = 100,
    random_state: int = 42,
):
    """function fit ALS

    Parameters
    ----------
    sparse_item_user : csr_matrix
        Сompressed Sparse Row matrix
    model_path: str
        Path to save model as pickle format
    iterations : int, optional
        Number of iterations, by default 12
    factors : int, optional
        Number of factors, by default 500
    regularization : float, optional
        Regularization, by default 0.2
    alpha : int, optional
        Alpha increments matrix values, by default 100
    random_state : int, optional
        Random state, by default 42

    Returns
    -------
    model: AlternatingLeastSquares
        trained model
    """
    model = implicit.als.AlternatingLeastSquares(factors=factors,
                                                 regularization=regularization,
                                                 iterations=iterations,
                                                 random_state=random_state)
    data_alpha = (sparse_item_user*alpha).astype('double')

    model.fit(data_alpha, show_progress=False)
    with open(model_path, "wb") as file:
        pickle.dump(model, file)
    return model


def load_model(
    model_path: str,
):
    """Function that load model from path

    Parameters
    ----------
    path : str
        Path to read model as pickle format

    Returns
    -------
    model: AlternatingLeastSquares
        Trained model
    """
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model


def personal_recomendations(
    user_id: int,
    n_similar: int,
    model: implicit.als.AlternatingLeastSquares,
    data: pd.DataFrame,
) -> List:
    """Give similar items from model

    Parameters
    ----------
    user_id : int
        User to whom we will recommend similar items
    n_similar : int
        Number of similar items
    model : als.AlternatingLeastSquares
        ALS model
    data : pd.DataFrame
        DataFrame containing streamer names & their ids

    Returns
    -------
    similar_items: List
        list of similar item to recomed user
    """
    # Get the item factors
    item_factors = model.item_factors

    # Normalize item factors
    item_norms = np.linalg.norm(item_factors, axis=1, keepdims=True)
    item_factors_normalized = item_factors / item_norms

    # Normalize user preferences
    user_preferences = model.user_factors[user_id]
    user_norm = np.linalg.norm(user_preferences)
    user_preferences_normalized = user_preferences / user_norm

    # Calculate the cosine similarity between the user and items
    similarity_scores = np.dot(item_factors_normalized, user_preferences_normalized)

    # Get the indices of the top N most similar items
    top_indices = np.argsort(similarity_scores)[::-1][:n_similar]
    # Map top_indices to original streamer_ids
    top_streamer_ids = data.iloc[top_indices]["streamer_id"].tolist()

    # Map streamer_ids to streamer_names using the provided data DataFrame
    similar_streamer_names = data[data["streamer_id"].isin(top_streamer_ids)]["streamer_name"].tolist()

    return similar_streamer_names



@app.get("/recomendations/user/{user_id}")
async def get_recomendation(user_id: int):
    """Fast Api Web Application

    Parameters
    ----------
    user_id : int
        user to whom we will recommend streamers

    Returns
    -------
    user: json
        user informations
    """
    data = process_data(path)
    model = load_model(model_path)

    personal = personal_recomendations(user_id, 100, model, data)

    user = User(user_id=user_id, personal=personal)
    return user


def main() -> None:
    """Run application"""
    uvicorn.run("solution:app", host="localhost")


if __name__ == "__main__":
    main()
