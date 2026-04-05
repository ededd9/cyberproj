import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

#function that removes links, numbers, etc from message/text
def clean_text(message):
    cleanedmessage = message.lower()
    cleanedmessage = re.sub(r'http\S+', 'url', cleanedmessage)
    cleanedmessage = re.sub(r'\d+','num', cleanedmessage)
    cleanedmessage = re.sub(r'[^a-z\s]', '',cleanedmessage)
    cleanedmessage = re.sub(r'\s+',' ', cleanedmessage)
    return cleanedmessage
#keep only v1 and v2 columns as thats all the important data in
df = pd.read_csv("./data/spam.csv", encoding="latin-1")
df = df[["v1","v2"]]
df.columns = ["label", "text"]
df["label"] = df["label"].map({"ham":0, "spam": 1})
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
df["clean_text"] = df["text"].apply(clean_text)
#convvert text to numbers
vectorizer = TfidfVectorizer(
    stop_words = "english",
    max_features = 5000,
    ngram_range=(1,2),
    min_df = 2,
    sublinear_tf=True
)
X = vectorizer.fit_transform(df["clean_text"])
y = df["label"]
#train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size = 0.2,
    random_state = 42,
    stratify=y
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples:  {X_test.shape[0]}")
#train and compare models 
models = {
    "Naive Bayes": MultinomialNB(alpha=0.1),
    "Logistic Regression": LogisticRegression(C=1.0, max_iter = 1000,class_weight="balanced"),
    "SVM": LinearSVC(C=1.0, class_weight="balanced", max_iter = 2000),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced") 
}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    recall = recall_score(y_test, preds)
    precision = precision_score(y_test,preds)
    f1 = f1_score(y_test, preds)
    print(f"{name:25} recall: {recall:.3f} precision {precision:.3f} f1: {f1:.3f}")
#threshold tuning on logistic regression
lr_model = models["Logistic Regression"]
probs = lr_model.predict_proba(X_test)[:,1]
print("\nLogistic Regression threshold tuning:")
for threshold in [0.5,0.4,0.3,0.2,0.1]:
    preds = (probs >= threshold).astype(int)
    recall = recall_score(y_test, preds)
    precision = precision_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    print(f"Threshold {threshold} → recall: {recall:.3f}  precision: {precision:.3f}  f1: {f1:.3f}")

#using model Logistic Regression and saving it to disk
final_model = models["Logistic Regression"]
with open("model.pkl", "wb") as f:
    pickle.dump(final_model,f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer,f)
print("Model and vectorizer saved.")
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
