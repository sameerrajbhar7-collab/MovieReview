import os
import re
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Preprocessing functions matching the notebook logic (with bug fix for stopwords to avoid empty texts)
def clean_data(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text) # Remove Punctuations 
    text = re.sub(r"http\S+", " ", text)    # Remove URLS
    text = re.sub(r"[<.*?>]", " ", text)    # Remove HTML tags style brackets
    text = text.strip().lower()             # Convert to Lowercase
    return text

def remove_stopwords(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    # Fix the bug in the original code where it replaced all tokens with empty strings
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)

def stemming(text):
    ps = PorterStemmer()
    tokens = word_tokenize(text)
    stemmed_words = [ps.stem(token) for token in tokens]
    return " ".join(stemmed_words)

# RNN Model architecture identical to the notebook
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

def main():
    dataset_path = r"C:\Users\pc\OneDrive\Music\Desktop\ML_Projects\Dataset\IMDB Dataset.csv"
    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    
    # Use a subset of the dataset for fast training while retaining good performance
    df = df.sample(n=15000, random_state=42)
    print(f"Using a sample of {len(df)} rows for training.")
    
    df.drop_duplicates(inplace=True)
    
    print("Preprocessing text...")
    df["review"] = df["review"].apply(clean_data)
    df["review"] = df["review"].apply(remove_stopwords)
    df["review"] = df["review"].apply(stemming)
    
    print("Encoding labels and vectorizing...")
    le = LabelEncoder()
    y = le.fit_transform(df["sentiment"])
    
    tf = TfidfVectorizer(max_features=5000)
    x = tf.fit_transform(df["review"]).toarray()
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    
    train_set = TensorDataset(
        torch.from_numpy(x_train).float(),
        torch.from_numpy(y_train).float()
    )
    test_set = TensorDataset(
        torch.from_numpy(x_test).float(),
        torch.from_numpy(y_test).float()
    )
    
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
    
    input_size = x_train.shape[1]
    model = RNN(input_size)
    
    criteria = nn.BCEWithLogitsLoss() # Use BCEWithLogitsLoss for numerical stability during training
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 5
    print(f"Training model for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            xb = xb.unsqueeze(1) # Add sequence length dimension (batch_size, 1, input_size)
            output = model(xb)
            loss = criteria(output.squeeze(), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(train_loader):.4f}")
        
    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.unsqueeze(1)
            output = model(xb)
            predicted = (torch.sigmoid(output.squeeze()) > 0.5).float()
            total += yb.size(0)
            correct += (predicted == yb).sum().item()
            
    print(f"Test Accuracy: {correct/total*100:.2f}%")
    
    # Save the assets
    print("Saving trained model and preprocessing objects...")
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/model.pth")
    with open("models/vectorizer.pkl", "wb") as f:
        pickle.dump(tf, f)
    with open("models/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    print("Training and serialization completed successfully!")

if __name__ == "__main__":
    main()
