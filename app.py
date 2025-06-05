import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Suppress TensorFlow logs

import tkinter as tk
from tkinter import messagebox
import joblib
import re
import numpy as np

# Load the pre-trained KNN model and TF-IDF vectorizer
model_dir = 'models'

# Load KNN model and vectorizer
knn_model = joblib.load(os.path.join(model_dir, 'knn_model.joblib'))
vectorizer = joblib.load(os.path.join(model_dir, 'tfidf_vectorizer.joblib'))  # Make sure you saved this!

def clean_text(text):
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return text.strip().lower()

def predict_sentiment():
    input_text = text_entry.get("1.0", tk.END).strip()
    if not input_text:
        messagebox.showwarning("Input Error", "Please enter some text!")
        return

    cleaned_text = clean_text(input_text)

    # Transform using the same TF-IDF vectorizer used during training
    text_tfidf = vectorizer.transform([cleaned_text])

    # Predict using KNN model
    prediction = knn_model.predict(text_tfidf)[0]

    result_label.config(text=f"Sentiment: {prediction}")

# GUI Setup
window = tk.Tk()
window.title("Sentiment Analysis App (KNN)")
window.geometry("400x300")

title_label = tk.Label(window, text="Enter Text for Sentiment Analysis", font=("Arial", 14))
title_label.pack(pady=10)

text_entry = tk.Text(window, height=10, width=40)
text_entry.pack(pady=10)

predict_button = tk.Button(window, text="Predict Sentiment", command=predict_sentiment)
predict_button.pack(pady=10)

result_label = tk.Label(window, text="Sentiment: ", font=("Arial", 12))
result_label.pack(pady=10)

window.mainloop()