import joblib
import os
"""
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'house_price_model.pkl')

# 모델을 한 번만 로드
model = joblib.load(MODEL_PATH)

def predict_house_price(area, rooms, floor):
    # ML 모델을 이용해 주택 매매가 예측
    X_input = [[area, rooms, floor]]
    predicted_price = model.predict(X_input)[0]
    return round(predicted_price, 2)
"""
