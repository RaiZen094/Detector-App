import pickle
from joblib import dump

# Load the .pkl file
with open("logistic_model.pkl", "rb") as file:
    model = pickle.load(file)

# Save the model as a .joblib file
dump(model, "logistic_model.joblib")
