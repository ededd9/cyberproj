import pickle
from cleaner import clean_text
from flask import Flask, render_template, request

app = Flask(__name__)
#load pre-trained model and the pre fitted vectorizer from disk : model has all its
#learned weights and vectorizer already knows the vocab / IDF scores from training
loaded_model = pickle.load(open("model.pkl", "rb"))
loaded_vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
# test_messages = ["Congratulations you won a free prize click here now","Please respond to the customers email ASAP","FREE BITCOIN IN 24 HOURS","Pizza party in the third floor tomorrow at 2:00 PM"]
# cleaned= []
# #list comprehension instead of for loop
# cleaned = [clean_text(message) for message in test_messages]
# vec = loaded_vectorizer.transform(cleaned)
# probs = loaded_model.predict_proba(vec)[:,1]
# label = []
# for message, prob in zip(test_messages,probs):
#     label = "THREAT" if prob >= 0.3 else "SAFE"
#     print(f"\nTest prediction [{label}] - {prob:.3f} confidence for message -> {message}")
@app.route("/", methods=["GET","POST"])
def home():
    result = None
    confidence = None
    message = None

    if request.method == "POST":
        #get the message
        message = request.form["message"]
        #clean the message
        cleaned = clean_text(message)
        #convert to numbers
        vec = loaded_vectorizer.transform([cleaned])
        #get threat probability
        prob = loaded_model.predict_proba(vec)[0][1]
        result = "THREAT" if prob >= 0.3 else "SAFE"
        if result == "THREAT":
            confidence = round(prob * 100, 1)           # threat probability
        else:
            confidence = round((1 - prob) * 100, 1)     # safe probability
    
    return render_template("index.html",result=result, confidence=confidence, message=message)
if __name__ == "__main__":
    app.run(debug=True)