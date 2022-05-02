import time
import joblib

class YouMLProject:
    def TextSentiment(str):
        svm_model = joblib.load('svm.joblib')
        return svm_model.predict([str])
