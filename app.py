import pickle
from cleaner import clean_text, analyze_paragraph
from flask import Flask, render_template, request
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

#SQLALCHEMY database config
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///cyberproj.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)
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

#database table
class History(db.Model):
    id = db.Column(db.Integer,primary_key=True)
    message = db.Column(db.String(200))
    result = db.Column(db.String(10))
    confidence = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
#create tables if none exist
with app.app_context():
    db.create_all()    
@app.route("/", methods=["GET","POST"])
def home():
    result = None
    confidence = None
    message = None
    threat_prob = None
    safe_prob = None
    sentence_results = []

    if request.method == "POST":
        #get the message
        message = request.form["message"]
        overall, max_prob, sentence_results = analyze_paragraph(message, loaded_model, loaded_vectorizer)
        result = overall
        confidence = round((max_prob if result == "THREAT" else 100 - max_prob),1)
        threat_prob = max_prob
        safe_prob = round(100-max_prob,1)    
        #write to db - stage entry and save to disk
        entry = History(message=message, result=result, confidence=confidence)
        db.session.add(entry)
        db.session.commit()

    
    history = History.query.order_by(History.timestamp.desc()).limit(10).all() 
    return render_template("index.html",result=result, confidence=confidence, message=message, threat_prob=threat_prob, safe_prob=safe_prob,sentence_results=sentence_results,history=history)
if __name__ == "__main__":
    app.run(debug=True)