import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib


df = pd.read_csv("dataset.csv")
texts = df["text"].astype(str).tolist()
labels = df["label"].apply(lambda x: 1 if x == "important" else 0).tolist()


model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModel.from_pretrained(model_name)
bert_model.eval()
print("✅ BERT loaded")

def get_embedding(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding.cpu().numpy()[0]

print("🔄 Encoding texts with BERT...")
embeddings = []
for txt in tqdm(texts):
    emb = get_embedding(txt)
    embeddings.append(emb)

X = np.vstack(embeddings)
y = np.array(labels)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
print("✅ Classifier trained")
print("Accuracy on train:", clf.score(X_train, y_train))
print("Accuracy on test:", clf.score(X_test, y_test))


joblib.dump(clf, "bert_email_clf.pkl")
tokenizer.save_pretrained("bert_tokenizer")
print("✅ Model and tokenizer saved")
