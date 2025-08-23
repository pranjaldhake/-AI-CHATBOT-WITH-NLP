import nltk
import numpy as np
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Ensure NLTK data is available ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer

# --- Sample Knowledge Base ---
intents = {
    "greeting": {
        "patterns": ["hello", "hi", "hey", "good morning", "good evening"],
        "responses": ["Hello!", "Hi there!", "Greetings!", "Hey! How can I help you today?"]
    },
    "goodbye": {
        "patterns": ["bye", "goodbye", "see you later", "farewell"],
        "responses": ["Goodbye!", "See you later!", "Take care!", "Have a great day!"]
    },
    "thanks": {
        "patterns": ["thanks", "thank you", "appreciate it"],
        "responses": ["You're welcome!", "No problem!", "Glad I could help!"]
    },
    "hours": {
        "patterns": ["what are your hours", "opening hours", "when are you open"],
        "responses": ["We're open from 9am to 5pm, Monday to Friday."]
    },
    "name": {
        "patterns": ["what is your name", "who are you", "your name"],
        "responses": ["I'm a simple chatbot created using NLTK!", "You can call me NLTK Bot."]
    },
}

lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in string.punctuation]
    return ' '.join(tokens)

def get_response(user_input):
    user_input = preprocess(user_input)

    corpus = []
    tag_lookup = []

    for tag, intent_data in intents.items():
        for pattern in intent_data['patterns']:
            corpus.append(preprocess(pattern))
            tag_lookup.append(tag)

    # Vectorize
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus + [user_input])

    # Cosine similarity
    similarity = cosine_similarity(X[-1], X[:-1])
    index = np.argmax(similarity)

    if similarity[0, index] < 0.2:
        return "I'm sorry, I didn't understand that."

    intent_tag = tag_lookup[index]
    return random.choice(intents[intent_tag]['responses'])

# --- Chat Loop ---
print("Chatbot: Hi! I'm your chatbot. Type 'quit' to exit.")

while True:
    user_input = input("You: ")
    if user_input.lower() in ['quit', 'exit', 'bye']:
        print("Chatbot:", random.choice(intents["goodbye"]["responses"]))
        break
    response = get_response(user_input)
    print("Chatbot:", response)
