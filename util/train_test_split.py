import pandas as pd
import numpy as np
from typing import Tuple
np.random.seed(0)


def split(
    num_users: int = None,
    num_test_items: int = 5,
    minimum_num_rating: int = None,
    df: pd.DataFrame = pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    df = df.groupby('User-ID').filter(lambda x: len(x["User-ID"]) >= 20)
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

    return train_df, test_df
