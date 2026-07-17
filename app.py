import os
import re
import pickle
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Preprocessing functions
def clean_data(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[<.*?>]", " ", text)
    text = text.strip().lower()
    return text

def remove_stopwords(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)

def stemming(text):
    ps = PorterStemmer()
    tokens = word_tokenize(text)
    stemmed_words = [ps.stem(token) for token in tokens]
    return " ".join(stemmed_words)

# RNN Model architecture matching train.py
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0) 
        out = self.fc(out[:, -1, :])
        return out

app = Flask(__name__)

# Global variables for model assets
model = None
vectorizer = None
label_encoder = None

def load_assets():
    global model, vectorizer, label_encoder
    model_path = "models/model.pth"
    vectorizer_path = "models/vectorizer.pkl"
    label_encoder_path = "models/label_encoder.pkl"
    
    if not (os.path.exists(model_path) and os.path.exists(vectorizer_path) and os.path.exists(label_encoder_path)):
        raise FileNotFoundError("Model assets not found. Please run train.py first.")
        
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
        
    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)
        
    input_size = len(vectorizer.get_feature_names_out())
    model = RNN(input_size)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    print("Model assets loaded successfully!")

@app.route("/", methods=["GET", "POST"])
def index():
    if model is None or vectorizer is None or label_encoder is None:
        try:
            load_assets()
        except Exception as e:
            return f"Error loading model assets: {str(e)}", 500

    sentiment = None
    review_text = ""

    if request.method == "POST":
        review_text = request.form.get("review", "")
        if review_text.strip():
            # Process review
            cleaned = clean_data(review_text)
            no_stop = remove_stopwords(cleaned)
            cleaned_text = stemming(no_stop)
            
            # Vectorize
            vectorized = vectorizer.transform([cleaned_text]).toarray()
            
            # Convert to Tensor
            tensor_input = torch.from_numpy(vectorized).float().unsqueeze(1)
            
            # Inference
            with torch.no_grad():
                output = model(tensor_input)
                probability = torch.sigmoid(output.squeeze()).item()
                
            # Get label prediction
            pred_label_idx = 1 if probability > 0.5 else 0
            sentiment = label_encoder.inverse_transform([pred_label_idx])[0]
            
    return render_template(
        "index.html", 
        sentiment=sentiment, 
        review_text=review_text
    )

if __name__ == "__main__":
    # Attempt to load assets on startup
    try:
        load_assets()
    except Exception as e:
        print(f"Warning on startup load: {e}. Make sure to run train.py.")
    app.run(debug=True, port=5000)
