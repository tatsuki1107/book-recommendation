import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
np.random.seed(0)


def split(
    num_test_items: int = 5,
    num_ratings_given_by_users: int = 15,
    num_items_rated: int = 10,
    df: pd.DataFrame = pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, List[int]]]:

    df = df.groupby(
        'User-ID').filter(lambda x: len(x["User-ID"]) >= num_ratings_given_by_users)
    df = df.reset_index(drop=True)
    df_index = list(df.index)
    unique_user_ids = df["User-ID"].unique()

    test_index = []
    for user_id in unique_user_ids:
        user_rating_item = list(df[df["User-ID"] == user_id].index)
        selected_index = np.random.choice(
            user_rating_item, num_test_items, replace=False
        )
        test_index.extend(selected_index)

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

    test_user_like_books = test_df[test_df["Book-Rating"] >=
                                   8].groupby('User-ID').agg({"ISBN": list})["ISBN"].to_dict()

    return train_df, test_df, test_user_like_books
