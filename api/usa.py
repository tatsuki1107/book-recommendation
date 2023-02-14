import joblib
from fastapi import HTTPException
from typing import List, Tuple
import pandas as pd
import numpy as np


def _open_pkl() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with open("mf-api.pkl", "rb") as f:
        user_df, book_df, rating_df = joblib.load(f)
    return user_df, book_df, rating_df


def _get_book_info(df: pd.DataFrame) -> List:
    columns = df.columns.to_list()
    book_info = []
    for i in range(len(df)):
        info = {}
        for column in columns:
            info[column] = df.iloc[i][column]

        book_info.append(info)

    return book_info


def get_recommend_list(user_id: int) -> List:

    user_df, book_df, rating_df = _open_pkl()

    user_df = user_df[user_df["user_id"] == user_id]
    index = user_df.index[0]
    recommended_book = user_df["mf_recommend_item"][index]

    book_df = book_df[book_df["book_id"].isin(recommended_book)]
    book_info = _get_book_info(book_df)

    return book_info


def get_history_list(user_id: int) -> List:

    user_df, book_df, rating_df = _open_pkl()

    rating_df = rating_df[rating_df["user_id"] == user_id]
    rating_df = rating_df.merge(book_df, on="book_id")

    rating_df = rating_df[rating_df.columns[rating_df.columns != 'user_id']]

    book_info = _get_book_info(rating_df)

    return book_info


def get_user_list() -> np.array:

    user_df, book_df, rating_df = _open_pkl()

    return user_df["user_id"].values
