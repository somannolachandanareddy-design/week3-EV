# Simple FAQ Chatbot using TF-IDF and cosine similarity
import sys
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Example FAQ corpus - extend as needed
faq_corpus = [
    "What is the project about?",
    "Which dataset is used?",
    "How do you predict energy consumption?",
    "What features are used in the model?",
    "How to run the Streamlit app?",
    "What are the peak charging hours?",
    "How accurate is the model?",
]

faq_answers = [
    "Predictive analysis of EV charging patterns using machine learning to forecast energy demand.",
    "We use the Electric Vehicle Charging Dataset from Kaggle, containing connection time, duration and energy consumed.",
    "We train regression models (Random Forest / Linear Regression) on features like Hour and ChargingDuration.",
    "Features: Hour, ChargingDuration. Target: EnergyConsumption.",
    "Run 'streamlit run streamlit_app.py' after training the model using train_model.py.",
    "From EDA, peak charging hours are usually between 5 PM and 9 PM.",
    "Model accuracy depends on dataset; check MAE and R2 from the training output."
]

vectorizer = TfidfVectorizer().fit_transform(faq_corpus)

print("EV Project Chatbot (type 'exit' to quit)")
while True:
    user = input('\nYou: ').strip()
    if user.lower() in ['exit','quit']:
        print('Goodbye!')
        break
    user_vec = TfidfVectorizer().fit(faq_corpus).transform([user])
    sims = cosine_similarity(user_vec, vectorizer).flatten()
    idx = sims.argsort()[-1]
    score = sims[idx]
    if score < 0.1:
        print("Bot: Sorry, I don't know the answer to that. Try asking about dataset, model, or how to run the app.")
    else:
        print('Bot:', faq_answers[idx])
