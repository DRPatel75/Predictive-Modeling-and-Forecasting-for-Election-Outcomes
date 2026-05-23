import joblib

state_encoder = joblib.load(r"D:\Projects\ML PROJECT 2\models\st_name_encoder.pkl")

print(state_encoder.classes_[:10])
print(type(state_encoder.classes_[0]))