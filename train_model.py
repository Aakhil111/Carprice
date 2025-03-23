import pickle

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

print("Expected Features Order:", model.feature_names_in_)
