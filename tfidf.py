from sklearn.feature_extraction.text import TfidfVectorizer


class TFIDF:
    def __init__(self, df, max_features=100, ngram_range=(1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = self.get_tfidf_vectorizer(df)

    def get_tfidf_vectorizer(self, df):
        vectoriser = TfidfVectorizer(ngram_range=self.ngram_range, max_features = self.max_features)
        vectoriser.fit(df)
        return vectoriser

    def df_tfidf_vectorize(self, X):
        print("starting vectorizing words...")
        X = self.vectorizer.transform(X)
        print("...words vectorized")
        return X