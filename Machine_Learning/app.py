from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

# Download NLTK data (if not already downloaded)
nltk.download('stopwords')
nltk.download('wordnet')

# Load the trained model
model = load_model('./ trained_model.h5', compile=False)

# Preprocessing functions (same as your code)
def cleaning_stopwords(text):
    STOPWORDS = set(stopwords.words('english'))
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

def cleaning_punctuations(text):
    punctuations_list = string.punctuation
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)

def cleaning_repeating_char(text):
    return re.sub(r'(.)\1+', r'\1', text)

def cleaning_email(text):
    return re.sub(r'@[^\s]+', ' ', text)

def cleaning_URLs(text):
    return re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))', ' ', text)

def cleaning_numbers(text):
    return re.sub(r'[0-9]+', '', text)

def preprocess_text(text):
    text = text.lower()
    text = cleaning_stopwords(text)
    text = cleaning_punctuations(text)
    text = cleaning_repeating_char(text)
    text = cleaning_email(text)
    text = cleaning_URLs(text)
    text = cleaning_numbers(text)

    tokenizer = RegexpTokenizer(r'\w+')
    text = tokenizer.tokenize(text)

    st = nltk.PorterStemmer()
    text = [st.stem(word) for word in text]

    lm = nltk.WordNetLemmatizer()
    text = [lm.lemmatize(word) for word in text]

    return " ".join(text)

# Tokenizer for converting text to sequences
max_len = 500
tok = Tokenizer(num_words=2000)

# Load or fit the tokenizer (ensure it matches the one used during training)
# tok.fit_on_texts(training_texts)

def predict_sentiment(text):
    preprocessed_text = preprocess_text(text)

    # Convert the text to a sequence
    sequence_input = tok.texts_to_sequences([preprocessed_text])
    padded_sequence = sequence.pad_sequences(sequence_input, maxlen=max_len)

    # Predict the sentiment
    prediction = model.predict(padded_sequence)
    prediction = (prediction > 0.5)  # Convert prediction to binary (0 or 1)

    if prediction == 1:
        return "positive"
    else:
        return "negative"

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define a route for sentiment analysis
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    sentiment = predict_sentiment(text)
    results = {
        "distribution": {
            sentiment: 100
        },
        "tweets": [
            {
                "id": "1",
                "text": "I love this product!",
                "sentiment": "positive",
                "score": 0.8,
            },
            {
                "id": "2",
                "text": "This is the worst experience ever.",
                "sentiment": "negative",
                "score": 0.1,
            },
            {
                "id": "3",
                "text": "Just bought a new phone.",
                "sentiment": "neutral",
                "score": 0.5,
            },
        ],
    }
    return jsonify(results)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)