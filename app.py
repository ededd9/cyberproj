import pickle
from cleaner import clean_text
#load pre-trained model and the pre fitted vectorizer from disk : model has all its
#learned weights and vectorizer already knows the vocab / IDF scores from training
loaded_model = pickle.load(open("model.pkl", "rb"))
loaded_vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
test_messages = ["Congratulations you won a free prize click here now","Please respond to the customers email ASAP","FREE BITCOIN IN 24 HOURS","Pizza party in the third floor tomorrow at 2:00 PM"]
cleaned= []
#list comprehension instead of for loop
cleaned = [clean_text(message) for message in test_messages]
vec = loaded_vectorizer.transform(cleaned)
probs = loaded_model.predict_proba(vec)[:,1]
label = []
for message, prob in zip(test_messages,probs):
    label = "THREAT" if prob >= 0.3 else "SAFE"
    print(f"\nTest prediction [{label}] - {prob:.3f} confidence for message -> {message}")