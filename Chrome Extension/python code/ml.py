import time
import joblib

class YouMLProject:
    def TextSentiment(str):
        ml_model = joblib.load('ml.joblib')
        return ml_model.predict([str])
