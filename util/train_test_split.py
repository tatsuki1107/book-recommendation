import joblib
import pandas as pd
import numpy as np
from .time import stop_watch
from typing import Tuple, Dict, List
from time import time
np.random.seed(0)


@stop_watch
def split(
    num_test_items: int = 5,
    num_ratings_given_by_users: int = 15,
    num_items_rated: int = 10,
    df: pd.DataFrame = pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    _df = df.copy()

    _df = _df.groupby(
        'user_id').filter(lambda x: len(x["user_id"]) >= num_ratings_given_by_users)
    _df = _df.reset_index(drop=True)
    df_index = _df.index.to_numpy()
    unique_user_ids = _df["user_id"].unique()

    test_index = []
    for user_id in unique_user_ids:
        user_rating_item = _df[_df["user_id"] == user_id].index.to_numpy()
        selected_index = np.random.choice(
            user_rating_item, num_test_items, replace=False
        )
        test_index.extend(selected_index)

    train_index = list(set(df_index) - set(test_index))

    train_df = _df.iloc[train_index]
    test_df = _df.iloc[test_index]

    train_df = train_df.groupby('book_id').filter(
        lambda x: len(x["book_id"]) >= num_items_rated)

    # trainデータのスコープ内で学習するのでtrainデータに出てくるアイテムとユーザをテストデータにする必要がある。
    train_user_ids = train_df["user_id"].unique()
    train_book_ids = train_df["book_id"].unique()
    test_df = test_df[test_df["user_id"].isin(train_user_ids)]
    test_df = test_df[test_df["book_id"].isin(train_book_ids)]

    user_df, book_df = _open_pkl()

    user_df = user_df[user_df["user_id"].isin(train_user_ids)]
    user_df = user_df.reset_index(drop=True)

    book_df = book_df[book_df["book_id"].isin(train_book_ids)]
    book_df = book_df.reset_index(drop=True)

    rating_df = pd.concat([train_df, test_df])

    return train_df, test_df, user_df, book_df, rating_df


def _open_pkl() -> Tuple[pd.DataFrame, pd.DataFrame]:
    with open("../data/user.pkl", "rb") as f:
        user_df = joblib.load(f)
    with open("../data/book.pkl", "rb") as f:
        book_df = joblib.load(f)

    return user_df, book_df
