from surprise.model_selection import KFold
from surprise import accuracy, Dataset, Reader, SVD
import numpy as np
import pandas as pd
from collections import defaultdict
import joblib
from util.time import stop_watch
pd.options.display.float_format = "{:.2f}".format


class MF:
    def __init__(self, train_df, test_df):
        reader = Reader(rating_scale=(1, 10))
        self.data_train = Dataset.load_from_df(train_df, reader)
        self.test_data = [tuple(e) for e in zip(
            test_df["User-ID"], test_df["ISBN"], test_df["Book-Rating"])]
        self.test_df = test_df

    def set_params(self, **kwargs):
        n_factors = kwargs.get("n_factors", 200)
        lr_all = kwargs.get("lr_all", 0.005)
        n_epochs = kwargs.get('n_epochs', 200)
        reg_all = kwargs.get('reg_all', 0.4)

        self.mf = SVD(
            n_factors=n_factors,
            lr_all=lr_all,
            n_epochs=n_epochs,
            reg_all=reg_all
        )

    @stop_watch
    def cross_validation(self):

        kf = KFold(n_splits=5)

        kf_p, kf_r, kf_rmse = [], [], []
        for train, val in kf.split(self.data_train):
            self.mf.fit(train)
            predictions = self.mf.test(val)
            precision, recall = self._precision_recall_at_k(predictions)

            kf_p.append(precision)
            kf_r.append(recall)
            kf_rmse.append(accuracy.rmse(predictions, verbose=False))

        # バリデーションでは、３指標の平均を返却値とする
        self.kfold_score = self._create_score_df(
            np.mean(kf_p), np.mean(kf_r), np.mean(kf_rmse))

    @stop_watch
    def test(self):
        full_data = self.data_train.build_full_trainset()
        self.mf.fit(full_data)
        self.test_bool = True
        predictions = self.mf.test(self.test_data)

        rmse = accuracy.rmse(predictions, verbose=False)
        precision, recall = self._precision_recall_at_k(predictions)

        self.test_score = self._create_score_df(precision, recall, rmse)

    @stop_watch
    def recommend(self, n=5):

        if not self.test_bool:
            raise("you have to call the test method")
        anti_testset = self.data_train.build_full_trainset().build_anti_testset(None)
        predictions = self.mf.test(anti_testset)

        test_ui_pair = set(tuple(e) for e in zip(
            self.test_df["User-ID"], self.test_df["ISBN"]))

        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            if (uid, iid) not in test_ui_pair:
                top_n[uid].append((iid, est))

        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = [d[0] for d in user_ratings[:n]]

        self.recommend_books = top_n

    def save_model(self):
        with open("../data/mf.pkl", "wb") as f:
            joblib.dump(self.mf, f)

    def reclist_to_csv(self):
        # if self.recommend_books is None:
        #    raise "you must run recommend method"
        #
        # with open("../data/explicit.pkl", "rb") as f:
        #    df = joblib.load(f)
        # with open("../data/user.pkl", "rb") as f:
        #    user_df = joblib.load(f)
        # with open("../data/book.pkl", "rb") as f:
        #    book_df = joblib.load(f)
        pass

    def _create_score_df(
        self,
        precision: float,
        recall: float,
        rmse: float
    ) -> pd.DataFrame:
        score_df = pd.DataFrame(
            data={
                "precision": precision,
                "recall": recall,
                "RMSE": rmse
            }, index=[1]
        )

        return score_df

    @stop_watch
    def _precision_recall_at_k(self, predictions, k=5, threshold=8):
        """Return precision and recall at k metrics for each user"""

        user_est_true = defaultdict(list)
        for uid, _, true_r, est, _ in predictions:
            user_est_true[uid].append((est, true_r))

        precisions = dict()
        recalls = dict()
        for uid, user_ratings in user_est_true.items():

            user_ratings.sort(key=lambda x: x[0], reverse=True)
            n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
            n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
            n_rel_and_rec_k = sum(
                ((true_r >= threshold) and (est >= threshold))
                for (est, true_r) in user_ratings[:k]
            )
            precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
            recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

        precision = sum(prec for prec in precisions.values()) / len(precisions)
        recall = sum(rec for rec in recalls.values()) / len(recalls)

        return precision, recall
