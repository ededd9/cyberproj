import pandas as pd
import re
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
print(df)
print(df.shape)
print(df["label"].value_counts())
df["clean_text"] = df["text"].apply(clean_text)
print(df)
print(df[["text", "clean_text"]].head(3))