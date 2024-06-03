from googleapiclient.discovery import build
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql import text
from flask import render_template
from google_auth_oauthlib.flow import InstalledAppFlow
from flask_migrate import Migrate
import sqlite3
import os
import pathlib
import email
import json

import requests
from flask import Flask, session, abort, redirect, request, url_for
from google.oauth2 import id_token
from google_auth_oauthlib.flow import Flow
from pip._vendor import cachecontrol
import google.auth.transport.requests
from google.oauth2.credentials import Credentials

#For summariser
import spacy
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import wordnet
nlp = spacy.load("en_core_web_sm")

import re

# Load a suitable sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

app = Flask("Google Login App")
# make sure this matches with that's in client_secret.json
app.secret_key = "STRING"

# to allow Http traffic for #local dev
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

GOOGLE_CLIENT_ID = "Write the string for Google client id, you can get it when you will enable the gmail api"

def create_flow(state = None):
    client_secrets_file = os.path.join(pathlib.Path(__file__).parent, "credentials.json")

    return Flow.from_client_secrets_file(client_secrets_file=client_secrets_file,
    scopes=["https://www.googleapis.com/auth/gmail.readonly", "https://www.googleapis.com/auth/userinfo.profile",
        "https://www.googleapis.com/auth/userinfo.email", "openid"],
        state = state,
    redirect_uri="http://localhost:5000/oauth2callback"
)

@app.route("/login")
def login():
    flow = create_flow()
    authorization_url, state = flow.authorization_url(prompt='consent')
    session["state"] = state
    return redirect(authorization_url)


def store_credentials(credentials):
    # Stores user credentials (access/refresh tokens) after authorization.
    # You may want to persist this using a file or database.

    # Simple example using Flask session
    session['credentials'] = credentials.to_json()


def fetch_credentials():
    # Retrieves stored user credentials, if present.
    return session.get('credentials')


@app.route("/logout")
def logout():
    session.pop('credentials', None)
    session.pop('state', None)
    return redirect("/")

@app.route("/")
def index():
    return render_template('home.html')

@app.route('/oauth2callback')
def oauth2callback():
    state = session.get("state")
    flow = create_flow(state = state)
    flow.fetch_token(authorization_response=request.url)

    credentials = flow.credentials
    store_credentials(credentials)
    return redirect("/inbox")

# Code below is further addition to the OAuth Flow Code, if you just want OAuth Flow, code above this line is fine.


app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///messages.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Optional
db = SQLAlchemy(app)
migrate = Migrate(app, db) 


class Message(db.Model):
    id = db.Column(db.String, primary_key=True)
    from_address = db.Column(db.String, index=True)  # Added index
    subject = db.Column(db.String, index=True)  # Added index
    # Added thread_id to accommodate threads of email.
    thread_Id = db.Column(db.String)
    body = db.Column(db.Text)  # Using 'Text' to accommodate potentially longer content
    conversation_count = db.Column(db.Integer, default=0) 

def get_gmail_service():

    credentials_dict = session.get('credentials')
    print("Credentials_Dict:", credentials_dict)
    print(type(credentials_dict))
    credentials_dict = json.loads(credentials_dict)

    if not credentials_dict:
        raise ValueError("No credentials available; user might not be logged in.")
    # Convert the credentials dictionary back to an OAuth2Credentials object
    credentials = Credentials(
        token=credentials_dict['token'],
        refresh_token=credentials_dict['refresh_token'],
        token_uri=credentials_dict['token_uri'],
        client_id=credentials_dict['client_id'],
        client_secret=credentials_dict['client_secret'],
        scopes=credentials_dict['scopes']
    )
    return build('gmail', 'v1', credentials=credentials)
import base64
import urllib.parse

def get_email_body(message_details):
    if 'parts' in message_details['payload']:
        for part in message_details['payload']['parts']:
            if part['mimeType'] == 'text/plain':
                return base64.urlsafe_b64decode(part['body']['data'].encode('ASCII')).decode('utf-8')
    return ""  # Default if no plain text body is found

# For Displaying threads.

        
@app.route('/inbox') 
def inbox():
    # ... (Your code to fetch messages) ...
    service = get_gmail_service()
    results = service.users().messages().list(userId='me').execute()
    messages = results.get('messages', [])
   
    # Prepare a list to store enhanced message data
    # enhanced_messages = []

    # While adding thread functionality,
    thread_dict = {}
    
    for message_id in messages:
    #Fetch message details
        message_details = service.users().messages().get(userId='me', id=message_id['id']).execute()
        thread_id = message_details.get('threadId')
        
        from_address = None
        subject = None
        
        #Extract from and subject from header
        for header in message_details['payload']['headers']:
            if header['name'] == 'From':
                from_address = header['value']
                
            if header['name'] == 'Subject':
                subject = header['value']
                
            email_body = get_email_body(message_details)
                
            message_data = {'id': message_id['id'], 'from_address': from_address, 'subject': subject,'body':email_body}
         
    # Pass the enhanced messages to the template
    # return render_template('inbox.html', messages=enhanced_messages)
        
        if thread_id:
            if thread_id in thread_dict:
                thread_dict[thread_id].append(message_data)
            else:
                thread_dict[thread_id] = [message_data]
    
        else:
            pass
    db.create_all()
    
    # Clear existing messages in the database
    db.session.query(Message).delete()
    
    for thread_id, combined_thread in thread_dict.items():
        # Concatenate messages within the thread
        concatenated_body = ""
        for msg in combined_thread:
            message = Message(id=msg['id'], from_address=msg['from_address'], 
                               subject=msg['subject'], thread_Id=thread_id,
                               body=msg['body'])
            db.session.add(message)
            
            Message.query.filter_by(thread_Id=thread_id).update({Message.conversation_count: Message.conversation_count + 1})
        # Store the concatenated representation
        #for msg in combined_thread:
             
    db.session.commit()
    
    page = request.args.get('page', 1, type=int)  
    messages_per_page = 20  # Adjust this as needed
    
    # Get unique threadIds (Modify this if you want to consider other factors,
    # like most recent message date, etc.)
    
    thread_ids = db.session.query(Message.thread_Id).distinct().paginate(page=page, per_page=messages_per_page)
    # Fetch and display thread details. Here I fetch all associated messages
    # but you might want to optimize this.
    threads = []
    for thread_id in thread_ids.items:
        messages = Message.query.filter_by(thread_Id=thread_id[0]).all()  
        # ... process the messages (e.g., extract a preview from the concatenated body)
        threads.append({
        'thread_id': thread_id, 'latest_subject': messages[0].subject,'latest_from_address': messages[0].from_address,
        'preview': messages[0].body[:100] + '...',  # A simple preview
        # ... other thread-related data (e.g., subject of the latest message)
    })
    
    next_url = url_for('inbox', page=thread_ids.next_num) if thread_ids.has_next else None
    prev_url = url_for('inbox', page=thread_ids.prev_num) if thread_ids.has_prev else None
    
    return render_template('inbox.html', threads=threads, next_url=next_url, prev_url=prev_url)
    
def get_messages_by_thread(thread_id):
    thread_id = thread_id[1:19:1]
    thread_id = thread_id.split("'")
    query_text = text("SELECT * FROM Message WHERE thread_Id=:thread_id") 
    result = db.session.execute(query_text, {"thread_id": thread_id[1]})
    return result.fetchall()
    
def get_messages(thread_id):
    query_text = text("SELECT * FROM Message WHERE thread_Id=:thread_id") 
    result = db.session.execute(query_text, {"thread_id": thread_id})
    return result.fetchall()
    
@app.route('/view_thread/<thread_id>')
def view_thread(thread_id):
    
    messages = get_messages_by_thread(thread_id) 
    thread_data = []
    for msg in messages:
        id = msg.thread_Id[0:18:1]
        id1= thread_id[1:19:1]
        id1 = id1.strip("'")
    
        if id1 == id:
            print(msg.thread_Id)
            thread_data.append({
            'from_address': msg.from_address,
            'subject': msg.subject,
            'body': msg.body,
            'thread_id':msg.thread_Id
            })
            break
    thread_data.reverse()
    return render_template('view_thread.html', thread_data=thread_data)

def preprocess_text(text):
    """Preprocess text by removing punctuation and lemmatizing."""
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_punct and not token.is_stop])

def calculate_importance(sentences, subject):
    """Calculate the importance of each sentence using topic modeling, TF-IDF, and embeddings."""
    # Preprocess sentences and subject
    processed_sentences = [preprocess_text(sentence.text) for sentence in sentences]
    processed_subject = preprocess_text(subject)

    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([processed_subject] + processed_sentences)
    subject_tfidf = tfidf_matrix[0]
    
    # Topic Modeling
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    topic_distribution = lda.fit_transform(tfidf_matrix)
    subject_topic = topic_distribution[0]

    # Sentence Embeddings
    embeddings = model.encode([processed_subject] + processed_sentences)
    subject_embedding = embeddings[0]
    
    # Calculate scores for each sentence
    importance_scores = []
    for i, sentence in enumerate(processed_sentences):
        # Cosine similarity for embeddings
        embedding_similarity = cosine_similarity([embeddings[i+1]], [subject_embedding])[0][0]
        
        # Cosine similarity for TF-IDF
#        tfidf_similarity = cosine_similarity([tfidf_matrix[i+1]], [subject_tfidf])[0][0]
        tfidf_similarity = cosine_similarity(tfidf_matrix[i+1], subject_tfidf)[0][0]

        
        # Similarity in topic distribution
        topic_similarity = cosine_similarity([topic_distribution[i+1]], [subject_topic])[0][0]
        
        # Combine the scores
        combined_score = 0.4 * embedding_similarity + 0.2 * tfidf_similarity + 0.4 * topic_similarity
        importance_scores.append(combined_score)
        

    return importance_scores
    
def preprocess_thread(text):
    """Preprocesses and removes angular brackets from a text document.

    Args:
        text (str): The input text string.

    Returns:
        str: The preprocessed text with individual angular brackets removed.
    """

    # Preprocessing step: Replace common HTML character encodings
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")

    # Regular expression to match individual angular brackets
    pattern = r"<|>"  

    # Remove all occurrences of the pattern
    clean_text = re.sub(pattern, "", text) 

    return clean_text
    
def summarise(text, subject, conversation_count, max_words = 100):
    """Generate a summary for the given text based on the calculated importance scores."""
    doc = nlp(text)
    sentences = list(doc.sents)
    scores = calculate_importance(sentences, subject)
    
    sorted_indices = np.argsort(scores)
    summary = []
    current_word_count = 0
    i = 0
    # Add sentences to summary until the word limit is reached or exceeded
    while sorted_indices.size > 0 and current_word_count < max_words:
        index = sorted_indices[-1]  # Take the most important remaining sentence
        sentence = sentences[index].text
        word_count = len(sentence.split())
        
        
        if current_word_count + word_count > max_words:
            # Check if adding this sentence would exceed the word limit
            if summary:  # If there's already some content, stop adding more
                break
            # If no sentences have been added yet, add the first one even if it's too long
            summary.append(sentence)
            break
        
        # Add the sentence to the summary and update the word count
        summary.append([])
        summary[i].append(sentence)
        summary[i].append(index)
        current_word_count += word_count
        sorted_indices = sorted_indices[:-1]  # Remove the last element (highest score)
        i = i+1
    summary = np.array(summary)

    column_index = 1
    indices = summary[:, 1].astype(int).argsort()
    
    sorted_summary = summary[indices]
        
    new_summary=[]

    for k in range(0,len(summary)):
        new_summary.append(sorted_summary[k][0])
    if(conversation_count > 1):
        return " ".join(new_summary[::-1])
    return " ".join(new_summary)


@app.route('/summarize',methods = ['GET'])
def summarize():
    message = request.args.get('thread_id')
    thread_data = get_messages(message)
    email_subject = thread_data[0][2]
    print(email_subject)
    email_body = thread_data[0][4]
    conversation_count = thread_data[0][5]
    email_body= preprocess_thread(email_body)
    print(email_body)

    summary = summarise(email_body, email_subject,conversation_count)
    print(summary)
    
    return render_template('summary.html', summary=summary)

if __name__ == "__main__":
    app.run(debug=True)
