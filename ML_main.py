# Save this code in a file called sensitivity_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report

class SensitivityModel:
    def __init__(self, data_file):
        df = pd.read_excel(data_file)
        X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['Label'], test_size=0.2, random_state=42)
        self.vectorizer = TfidfVectorizer()
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        self.model = svm.SVC(kernel='linear')
        self.model.fit(X_train_tfidf, y_train)
        predictions = self.model.predict(X_test_tfidf)
        print(classification_report(y_test, predictions))

    def predict_sensitivity(self, text):
        text_tfidf = self.vectorizer.transform([text])
        prediction = self.model.predict(text_tfidf)
        return prediction



model = SensitivityModel('training and testing/training_data.xlsx')