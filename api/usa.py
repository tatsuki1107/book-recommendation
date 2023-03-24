from exception import logging_exception
import joblib
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np

@logging_exception
def _open_pkl() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with open("mf-api.pkl", "rb") as f:
        user_df, book_df, rating_df = joblib.load(f)
    return user_df, book_df, rating_df

@logging_exception
def _get_book_info(df: pd.DataFrame) -> List:
    columns = df.columns.to_list()
    book_info = []
    for i in range(len(df)):
        info = {}
        for column in columns:
            if column == "title_embedding":
                continue
            info[column] = df.iloc[i][column]

        book_info.append(info)

    return book_info

@logging_exception
def _get_author_publisher_info(column: str, df: pd.DataFrame) -> List:
    df = df[df["book_rating"] >= 8.0]
    if len(df) == 0:
        return []

    df = df.groupby(column)["book_rating"].agg(
        {"mean", "count"}).sort_values(by=["mean", "count"], ascending=False)

    target_list = df.index.to_list()
    counts = df["count"].to_list()
    means = df["mean"].to_list()

    column_rating_info = []
    for target, count, mean in zip(target_list, counts, means):
        _dict = {}
        _dict[column], _dict["count"], _dict["mean"] = target, count, mean
        column_rating_info.append(_dict)

    return column_rating_info

@logging_exception
def get_recommend_list(user_id: int) -> List:

    user_df, book_df, rating_df = _open_pkl()

    user_df = user_df[user_df["user_id"] == user_id]
    index = user_df.index[0]
    recommended_book = user_df["mf_recommend_item"][index]

    book_df = book_df[book_df["book_id"].isin(recommended_book)]
    book_info = _get_book_info(book_df)

    return book_info

@logging_exception
def get_history_list(user_id: int) -> Dict:

    user_df, book_df, rating_df = _open_pkl()

    rating_df = rating_df[rating_df["user_id"] == user_id]
    rating_df = rating_df.merge(book_df, on="book_id")

    rating_df = rating_df[rating_df.columns[rating_df.columns != 'user_id']]
    book_info = _get_book_info(rating_df)

    history_info = dict()
    for column in ["book_author", "publisher"]:
        column_rating_info = _get_author_publisher_info(column, rating_df)
        history_info[f"rated_{column}"] = column_rating_info

    history_info["history_book"] = book_info

    return history_info

@logging_exception
def get_user_list() -> np.array:

    user_df, book_df, rating_df = _open_pkl()

    return user_df["user_id"].values
