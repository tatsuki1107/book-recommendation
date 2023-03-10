import pandas as pd
import numpy as np
from .time import stop_watch
from typing import Tuple, Dict, List
from time import time
np.random.seed(0)
import joblib


@stop_watch
def split(
    num_test_items: int = 5,
    num_ratings_given_by_users: int = 15,
    num_items_rated: int = 10,
    df: pd.DataFrame = pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, List[int]]]:

    df = df.groupby(
        'User-ID').filter(lambda x: len(x["User-ID"]) >= num_ratings_given_by_users)
    df = df.reset_index(drop=True)
    df_index = df.index.to_numpy()
    unique_user_ids = df["User-ID"].unique()

    test_index = []
    start = time()
    for user_id in unique_user_ids:
        user_rating_item = df[df["User-ID"] == user_id].index.to_numpy()
        selected_index = np.random.choice(
            user_rating_item, num_test_items, replace=False
        )
        test_index.extend(selected_index)
    stop = time()
    print(f"for文のタイム: {stop-start:.2f}")

    train_index = list(set(df_index) - set(test_index))

    train_df = df.iloc[train_index]
    test_df = df.iloc[test_index]

    train_df = train_df.groupby('ISBN').filter(
        lambda x: len(x["ISBN"]) >= num_items_rated)

    # trainデータのスコープ内で学習するのでtrainデータに出てくるアイテムとユーザをテストデータにする必要がある。
    train_user_ids = train_df["User-ID"].unique()
    train_book_ids = train_df["ISBN"].unique()
    test_df = test_df[test_df["User-ID"].isin(train_user_ids)]
    test_df = test_df[test_df["ISBN"].isin(train_book_ids)]

    return train_df, test_df

def _open_pkl():
    with open("../data/user.pkl", "rb") as f:
        user_df = joblib.load(f)
    with open("../data/book.pkl", "rb") as f:
        book_df = joblib.load(f)
    
    return user_df, book_df
        
        