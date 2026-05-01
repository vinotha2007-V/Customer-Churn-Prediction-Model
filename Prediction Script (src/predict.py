import pickle
import numpy as np

def predict(data):
    with open('models/churn_model.pkl', 'rb') as f:
        model = pickle.load(f)

    prediction = model.predict([data])
    return prediction

if __name__ == "__main__":
    sample = np.random.rand(19)
    print("Prediction:", predict(sample))
