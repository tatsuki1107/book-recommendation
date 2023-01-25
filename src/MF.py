import numpy as np
import pandaas as pd
from collections import defaultdict
from util.models import RecommendResult
from surprise import SVD, Reader, Dataset
np.random.seed(0)


class MFRecommender:
    def recommend(self, train_df, test_df, **kwargs):
        # 因子数
        factors = kwargs.get("factors", 5)
        # 評価数の閾値
        minimum_num_rating = kwargs.get("minimum_num_rating", 100)
        # バイアス項の使用
        use_biase = kwargs.get("use_biase", False)
        # 学習率
        lr_all = kwargs.get("lr_all", 0.005)
        # エポック数
        n_epochs = kwargs.get("n_epochs", 50)

        # Surprise用にデータを加工
        reader = Reader(rating_scale=(1, 10))
        data_train = Dataset.load_from_df(
            train_df[["User-ID", "ISBN", "Book-Rating"]], reader
        ).build_full_trainset()

        # Surpriseで行列分解を学習
        # SVDという名前だが、特異値分解ではなく、Matrix Factorizationが実行される
        matrix_factorization = SVD(
            n_factors=factors,
            n_epochs=n_epochs,
            lr_all=lr_all,
            biased=use_biase
        )
        matrix_factorization.fit(data_train)

        def get_top_n(predictions, n=10):
            # 各ユーザーごとに、予測されたアイテムを格納する
            top_n = defaultdict(list)
            for uid, iid, true_r, est, _ in predictions:
                top_n[uid].append((iid, est))

            # ユーザーごとに、アイテムを予測評価値順に並べ上位n個を格納する
            for uid, user_ratings in top_n.items():
                user_ratings.sort(key=lambda x: x[1], reverse=True)
                top_n[uid] = [d[0] for d in user_ratings[:n]]

            return top_n

        # 学習データに出てこないユーザーとアイテムの組み合わせを準備
        data_test = data_train.build_anti_testset(None)
        predictions = matrix_factorization.test(data_test)
        pred_user2items = get_top_n(predictions, n=10)

        test_data = pd.DataFrame.from_dict(
            [{"User-ID": p.uid, "ISBN": p.iid, "rating_pred": p.est}
                for p in predictions]
        )
        book_rating_predict = test_df.merge(
            test_data, on=["user_id", "movie_id"], how="left")

        # 予測ができない箇所には、平均値を格納する
        book_rating_predict.rating_pred.fillna(
            train_df.rating.mean(), inplace=True)

        return RecommendResult(book_rating_predict.rating_pred, pred_user2items)
