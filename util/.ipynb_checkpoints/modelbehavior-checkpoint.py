import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from itertools import chain
from typing import Dict, Set
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
sns.set()


def check_over_fit(user_df: pd.DataFrame, book_df: pd.DataFrame, rating_df: pd.DataFrame) -> None:
    """評価された数が少なく、平均的に高評価なアイテムに過学習しているかチェック"""

    if user_df["mf_recommend_item"] is None:
        raise "you must run recommend method"

    recommended_books = user_df["mf_recommend_item"].to_list()
    recommended_books = list(chain.from_iterable(recommended_books))

    book_count = {}
    for book in recommended_books:
            book_count[book] = 1 + book_count.get(book, 0)

    print(f"各ユーザにレコメンドされた本のユニーク数: {len(book_count)}")

    book_count = dict(sorted(book_count.items(), key=lambda x: x[0]))

    rec_df = rating_df.copy()

    rec_df = rec_df[rec_df["book_id"].isin(set(recommended_books))].groupby("book_id").agg(
            {"book_rating": ["mean", "count"]}).sort_values(by="book_id")
    rec_df["recommend_count"] = list(book_count.values())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))

    sns.scatterplot(rec_df, x=("book_rating", "count"),
                        y="recommend_count", ax=ax1)
    sns.scatterplot(rec_df, x=("book_rating", "mean"),
                        y="recommend_count", ax=ax2)

    del rec_df

def probability_rated_book_info(rating_df: pd.DataFrame, book_df: pd.DataFrame, user_df: pd.DataFrame) -> None:
    """過去に高評価した著者、出版社の作品がレコメンドリスト内に含まれている確率を出すメソッド"""

    high_rating_df = rating_df.copy()
    high_rating_df = high_rating_df[high_rating_df["book_rating"] >= 8.0]
    high_rating_df = high_rating_df.merge(book_df, on="book_id")
        
    rec_df = user_df.copy()
    rec_df = rec_df[["user_id", "mf_recommend_item"]]

    def convert_to_dict_and_calc_prob(column: str) -> dict:
        rated_column = high_rating_df.groupby('user_id').agg({column: set})[column].to_dict()

        def replace_book_id_to_column_value(x) -> Set:
            res = []
            for book_id in x:
                if len(book_df[book_df["book_id"] == book_id]) == 0:
                    continue
                value = self.book_df[self.book_df["book_id"] == book_id][column].to_list()[0]
                res.append(value)

            return set(res)

        rec_df["mf_recommend_item"] = rec_df["mf_recommend_item"].map(
            replace_book_id_to_column_value)

        recommended_column = dict(zip(rec_df["user_id"], rec_df["mf_recommend_item"]))

        p = dict()
        for user_id in high_rating_df["user_id"].unique():
            p[user_id] = len(rated_column[user_id] & recommended_column[user_id]) / 5
            value_p = round(sum(_p for _p in p.values()) / len(p), 3)

            return value_p

        prob = dict()
        for column in ["book_author", "publisher"]:
            value_p = convert_to_dict_and_calc_prob(column)
            prob[f"高評価した{column}がレコメンドされている確率"] = value_p

        del high_rating_df, rec_df

        return prob

def plot_pca(qi: np.ndarray) -> None:
    """アイテムの因子行列を主成分分析することで過剰に反応している本がないかチェックする。"""

    scaler = StandardScaler()
    qi = scaler.fit_transform(qi)

    pca = PCA()
    pca.fit(qi)
    qi = pca.transform(qi)

    contribution_ratios = pd.DataFrame(pca.explained_variance_ratio_)
    cumulative_contribution_ratios = contribution_ratios.cumsum()

    cont_cumcont_ratios = pd.concat([contribution_ratios, cumulative_contribution_ratios], axis=1).T
    cont_cumcont_ratios.index = ['contribution_ratio', 'cumulative_contribution_ratio']

    x_axis = range(1, contribution_ratios.shape[0] + 1)
    plt.rcParams['font.size'] = 18

    plt.bar(x_axis, contribution_ratios.iloc[:, 0], align='center')
    plt.plot(x_axis, cumulative_contribution_ratios.iloc[:, 0], 'r.-')

    plt.xlabel('Number of principal components')
    plt.ylabel('Contribution ratio(blue),\nCumulative contribution ratio(red)')
    plt.show()
    
def plot_heatmap(matrix: np.ndarray) -> None:
    """評価値matrixをヒートマップで可視化
    args:
          matrix: 評価値行列
    """
    fig, ax = plt.subplots(figsize=(20,5))

    my_cmap = plt.cm.get_cmap('Reds')
    heatmap = plt.pcolormesh(matrix.T, cmap=my_cmap)
    colorbar = plt.colorbar(heatmap)
    ax.grid()
    plt.tight_layout()
    plt.show()
