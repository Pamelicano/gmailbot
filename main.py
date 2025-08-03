import os
import base64
import joblib
import torch
from transformers import AutoTokenizer, AutoModel
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.auth.transport.requests import Request

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

tokenizer = AutoTokenizer.from_pretrained("bert_tokenizer")
bert_model = AutoModel.from_pretrained("bert-base-multilingual-cased")
bert_model.eval()
clf = joblib.load("bert_email_clf.pkl")

def authenticate():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            raise Exception("No valid credentials. Run quickstart first.")
    return build('gmail', 'v1', credentials=creds)

def get_subject(headers):
    for h in headers:
        if h['name'].lower() == 'subject':
            return h['value']
    return "(no subject)"

def get_embedding(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding.cpu().numpy()

def classify_email(text: str):
    emb = get_embedding(text)
    pred = clf.predict(emb)[0]
    prob = clf.predict_proba(emb)[0][pred]
    return ("important" if pred == 1 else "not_important", float(prob))

def get_recent_emails(service, max_results=5):
    results = service.users().messages().list(userId='me', maxResults=max_results).execute()
    messages = results.get('messages', [])
    emails = []
    for msg in messages:
        txt = service.users().messages().get(userId='me', id=msg['id'], format='full').execute()
        payload = txt['payload']
        headers = payload.get('headers', [])
        subject = get_subject(headers)

        parts = payload.get('parts', [])
        data = ''
        if 'body' in payload and 'data' in payload['body']:
            data = payload['body']['data']
        else:
            for part in parts:
                if 'body' in part and 'data' in part['body']:
                    data = part['body']['data']
                    break
        body_text = ""
        if data:
            body_text = base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
        emails.append((subject, body_text))
    return emails

if __name__ == '__main__':
    print("Loading model and tokenizer...")
    print("Connecting to Gmail API...")
    service = authenticate()
    print("Fetching recent emails...")
    emails = get_recent_emails(service, max_results=5)
    print(f"Fetched {len(emails)} emails.")
    for subject, email_text in emails:
        label, conf = classify_email(subject)
        print(f"Subject: {subject} | Predicted: {label} | confidence: {conf:.2f}")
