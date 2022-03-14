from tfidf_models import TFIDFLinearSVC
import pickle
from preprocessing import Preprocessing
from model_selection import ModelSelection


class Train:
    def train_model(self):
        prp = Preprocessing()
        df = prp.main()
        df = Preprocessing.preprocess(df)
        X_train, X_test, Y_train, Y_test = ModelSelection.df_train_test_split(df, "text", "sentiment", test_size=0.05)
        model_json = {
            "model": "TFIDFLinearSVC",
            "model__tfidf_max_features": 8739130,
            "model__penalty": "l1",
            "model__dual": False,
            "model__tol": 0.000818,
            "model__C": 0.433223,
            "model__intercept_scaling": 9.470073,
            "model__max_iter": 549
        }

        model = TFIDFLinearSVC(
            tfidf_max_features=model_json["model__tfidf_max_features"],
            penalty=model_json["model__penalty"],
            dual=model_json["model__dual"],
            tol=model_json["model__tol"],
            C=model_json["model__C"],
            intercept_scaling=model_json["model__intercept_scaling"],
            max_iter=model_json["model__max_iter"] + 100,
        )

        model.fit(X_train, Y_train)
        with open('TFIDF_linearSVC_test.pkl', 'wb') as fid:
            pickle.dump(model, fid)


t = Train()
t.train_model()

