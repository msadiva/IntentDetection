from dotenv import load_dotenv
import os
import joblib
import openai
import numpy as np

print(np.__version__)

load_dotenv()


# Load model and encoder
try:
    clf = joblib.load("model.joblib")
except Exception as e:
    print(f"Error loading model: {e}")

try:
    label_encoder = joblib.load("label_encoder.joblib")
except Exception as e:
    print(f"Error loading label encoder: {e}")

# Embedding model
EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def get_embedding(text: str, model=EMBEDDING_MODEL) -> list:
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.embeddings.create(model=model, input=text)
    return response.data[0].embedding


def predict_label(text: str) -> str:
    embedding = get_embedding(text)
    prediction = clf.predict([embedding])[0]
    label = label_encoder.inverse_transform([prediction])[0]
    return label
