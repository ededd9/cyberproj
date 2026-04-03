import pandas as pd
import numpy as np
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score, precision_score, f1_score

#==========================#
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier



#==========================#
#function that removes links, numbers, etc from message/text
def clean_text(message):
    cleanedmessage = message.lower()
    cleanedmessage = re.sub(r'http\S+', 'url', cleanedmessage)
    cleanedmessage = re.sub(r'\d+','num', cleanedmessage)
    cleanedmessage = re.sub(r'[^a-z\s]', '',cleanedmessage)
    cleanedmessage = re.sub(r'\s+',' ', cleanedmessage)
    return cleanedmessage
df = pd.read_csv("./data/spam.csv", encoding="latin-1")
#keep only v1 and v2 columns as thats all the important data in
df = df[["v1","v2"]]
df.columns = ["label", "text"]
df["label"] = df["label"].map({"ham":0, "spam": 1})
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
df["clean_text"] = df["text"].apply(clean_text)

vectorizer = TfidfVectorizer(
    stop_words = "english",
    max_features = 5000,
    ngram_range=(1,2),
    min_df = 2,
    sublinear_tf=True
)

X = vectorizer.fit_transform(df["clean_text"])
y = df["label"]

print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size = 0.2,
    random_state = 42,
    stratify=y
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples:  {X_test.shape[0]}")
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

lr_model = models["Logistic Regression"]
probs = lr_model.predict_proba(X_test)[:,1]
print("\nLogistic Regression threshold tuning:")
for threshold in [0.5,0.4,0.3,0.2,0.1]:
    preds = (probs >= threshold).astype(int)
    recall = recall_score(y_test, preds)
    precision = precision_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    print(f"Threshold {threshold} → recall: {recall:.3f}  precision: {precision:.3f}  f1: {f1:.3f}")
# model = MultinomialNB(alpha=0.1)
# model.fit(X_train, y_train)
# print("Model trained")

# predictions = model.predict(X_test)
# print(classification_report(y_test,predictions,target_names=["Safe","Threat"]))

#test different thresholds
# probs = model.predict_proba(X_test)[:, 1]
# for threshold in [0.5,0.4,0.3,0.2]:
#     preds = (probs >= threshold).astype(int)
#     recall = recall_score(y_test, preds)
#     precision = precision_score(y_test,preds)
#     print(f"Threshold {threshold} -> recall: {recall:.3f} precision {precision:.3f}")
