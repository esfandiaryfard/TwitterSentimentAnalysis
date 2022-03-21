from sklearn.model_selection import train_test_split
import json
from settings import BASE_DIR
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
import scipy.stats.distributions as dists
import tfidf_models as cnb
import w2v_models as w2v
from preprocessing import Preprocessing


class ModelSelection:
    def df_train_test_split(df, var_text, var_sentiment, test_size):
        print("starting splitting dataset...")
        X = df[var_text]
        Y = df[var_sentiment]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
        print("...dataset splitted")
        return X_train, X_test, Y_train, Y_test

    def save_results(search, file_name):
        search = search.cv_results_
        for i in range(len(search['params'])):
            model = search['params'][i]
            model['model'] = str(model['model'])
            model['model'] = model['model'][:model['model'].find('(')]
            model['mean_test_score'] = search['mean_test_score'][i]
            model['mean_score_time'] = search['mean_score_time'][i]
            model['mean_fit_time'] = search['mean_fit_time'][i]
            model["mean_train_score"] = search['mean_train_score'][i]
            with open("{BaseDir}/TwiiterSentimentAnalysis/{FileName}.json".format(BaseDir=BASE_DIR, FileName=file_name),
                      'a') as f:
                json.dump(model, f, indent=4)

    def apply_random_search(pip, params_grid, X_train, Y_train, results_file_name):
        for i in range(50):
            trys = RandomizedSearchCV(pip, param_distributions=params_grid, n_iter=1, n_jobs=4, return_train_score=True)
            search = trys.fit(X_train, Y_train)
            ModelSelection.save_results(search, results_file_name)

    def first_screening(self):
        prp = Preprocessing()
        df = prp.main()
        df = Preprocessing.resize(df, 50000, "sentiment", 4)
        df = Preprocessing.preprocess(df)
        X_train, X_test, Y_train, Y_test = ModelSelection.df_train_test_split(df, "text", "sentiment", test_size=0.05)
        w2v1 = w2v.W2VDecisionTreeClassifier()
        pip = Pipeline([('model', w2v1)])
        params_grid = [dict(model=[w2v.W2VDecisionTreeClassifier()],
                            model__vector_size=dists.randint(1, 10000 + 1),
                            model__window=dists.randint(1, 10 + 1),
                            model__min_count=dists.randint(1, 10 + 1),
                            model__epochs=dists.randint(5, 25 + 1),
                            model__criterion=['gini', 'entropy'],
                            model__splitter=['best', 'random'],
                            model__max_depth=dists.randint(1, 10 + 1),
                            model__min_samples_split=dists.randint(1, 20 + 1),
                            model__min_samples_leaf=dists.randint(1, 20 + 1),
                            model__min_weight_fraction_leaf=dists.uniform(0, 0.5),
                            model__max_features=[None, "auto", "sqrt", "log2"],
                            model__max_leaf_nodes=dists.randint(0, 1000 + 1),
                            model__min_impurity_decrease=dists.uniform(0.0, 10.0),
                            model__ccp_alpha=dists.uniform(0.0, 10.0)
                            ),
                       dict(model=[w2v.W2VGaussianNB()],
                            model__vector_size=dists.randint(1, 10000 + 1),
                            model__window=dists.randint(1, 10 + 1),
                            model__min_count=dists.randint(1, 10 + 1),
                            model__epochs=dists.randint(5, 25 + 1)
                            ),
                       dict(model=[w2v.W2VLinearSVC()],
                            model__vector_size=dists.randint(1, 10000 + 1),
                            model__window=dists.randint(1, 10 + 1),
                            model__min_count=dists.randint(1, 10 + 1),
                            model__epochs=dists.randint(5, 25 + 1),
                            model__penalty=['l1', 'l2'],
                            model__dual=[False],
                            model__tol=dists.uniform(0.00001, 0.001),
                            model__C=dists.uniform(0.1, 2.0),
                            model__intercept_scaling=dists.uniform(0.1, 10),
                            model__max_iter=dists.randint(100, 2000 + 1)
                            ),
                       dict(model=[w2v.W2VMultinomialNB()],
                            model__vector_size=dists.randint(1, 10000 + 1),
                            model__window=dists.randint(1, 10 + 1),
                            model__min_count=dists.randint(1, 10 + 1),
                            model__epochs=dists.randint(5, 25 + 1),
                            model__alpha=dists.uniform(0, 5.0)
                            ),
                       dict(model=[w2v.W2VLogisticRegression()],
                            model__vector_size=dists.randint(1, 10000 + 1),
                            model__window=dists.randint(1, 10 + 1),
                            model__min_count=dists.randint(1, 10 + 1),
                            model__epochs=dists.randint(5, 25 + 1),
                            model__penalty=['l1', 'l2', 'elasticnet', 'none'],
                            model__tol=dists.uniform(0.00001, 0.001),
                            model__C=dists.uniform(0.1, 2.0),
                            model__fit_intercept=[True, False],
                            model__solver=['saga'],
                            model__max_iter=dists.randint(100, 1000 + 1),
                            model__l1_ratio=dists.uniform(0.0, 1.0)
                            ),
                       dict(model=[cnb.TFIDFDecisionTreeClassifier()],
                            model__tfidf_max_features=dists.randint(1, 50000000),
                            model__ngram_range=[(1, 2)],
                            model__criterion=['gini', 'entropy'],
                            model__splitter=['best', 'random'],
                            model__max_depth=dists.randint(1, 10 + 1),
                            model__min_samples_split=dists.randint(2, 20 + 1),
                            model__min_samples_leaf=dists.randint(1, 20 + 1),
                            model__min_weight_fraction_leaf=dists.uniform(0, 0.5),
                            model__max_features=[None, "auto", "sqrt", "log2"],
                            model__max_leaf_nodes=dists.randint(0, 1000 + 1),
                            model__min_impurity_decrease=dists.uniform(0.0, 10.0),
                            model__ccp_alpha=dists.uniform(0.0, 10.0)
                            ),
                       dict(model=[cnb.TFIDFGaussianNB()],
                            model__tfidf_max_features=dists.randint(1, 50000000),
                            model__ngram_range=[(1, 2)]
                            ),
                       dict(model=[cnb.TFIDFLinearSVC()],
                            model__tfidf_max_features=dists.randint(1, 50000000),
                            model__ngram_range=[(1, 2)],
                            model__penalty=['l1', 'l2'],
                            model__dual=[False],
                            model__tol=dists.uniform(0.00001, 0.001),
                            model__C=dists.uniform(0.1, 2.0),
                            model__intercept_scaling=dists.uniform(0.1, 10),
                            model__max_iter=dists.randint(100, 2000 + 1)
                            ),
                       dict(model=[cnb.TFIDFMultinomialNB()],
                            model__tfidf_max_features=dists.randint(1, 50000000),
                            model__ngram_range=[(1, 2)],
                            model__alpha=dists.uniform(0, 5.0)
                            ),
                       dict(model=[cnb.TFIDFLogisticRegression()],
                            model__tfidf_max_features=dists.randint(1, 50000000),
                            model__ngram_range=[(1, 2)],
                            model__penalty=['l1', 'l2', 'elasticnet', 'none'],
                            model__tol=dists.uniform(0.00001, 0.001),
                            model__C=dists.uniform(0.1, 2.0),
                            model__fit_intercept=[True, False],
                            model__solver=['saga'],
                            model__max_iter=dists.randint(100, 1000 + 1),
                            model__l1_ratio=dists.uniform(0.0, 1.0)
                            )]

        ModelSelection.apply_random_search(pip, params_grid, X_train, Y_train, "RandomSearchModelResults")

    def second_screening(self):
        prp = Preprocessing()
        df = prp.main()
        df = Preprocessing.preprocess(df)
        X_train, X_test, Y_train, Y_test = ModelSelection.df_train_test_split(df, "text", "sentiment", test_size=0.05)

        cnb1 = cnb.TFIDFLinearSVC()

        pip = Pipeline([('model', cnb1)])
        params_grid = [dict(model=[cnb.TFIDFLinearSVC()],
                            model__tfidf_max_features=dists.randint(6614037, 44901229),
                            model__penalty=['l1', 'l2'],
                            model__dual=[False],
                            model__tol=dists.uniform(0.000014, 0.00099),
                            model__C=dists.uniform(0.3, 2.0),
                            model__intercept_scaling=dists.uniform(0.22, 9.51),
                            model__max_iter=dists.randint(177, 1706)
                            ),
                       dict(model=[cnb.TFIDFMultinomialNB()],
                            model__tfidf_max_features=dists.randint(722114, 49558100),
                            model__alpha=dists.uniform(0, 4.90)
                            )]

        ModelSelection.apply_random_search(pip, params_grid, X_train, Y_train, "second_screening_results")


