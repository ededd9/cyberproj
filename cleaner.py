import re
def clean_text(message):
    cleanedmessage = message.lower()
    cleanedmessage = re.sub(r'http\S+', 'url', cleanedmessage)
    cleanedmessage = re.sub(r'\d+','num', cleanedmessage)
    cleanedmessage = re.sub(r'[^a-z\s]', '',cleanedmessage)
    cleanedmessage = re.sub(r'\s+',' ', cleanedmessage)
    return cleanedmessage
def analyze_paragraph(text,model,vectorizer,threshold=0.3):
    # split text into sentences on . ! or ?
    sentences = re.split(r'[.!?]+', text)

    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    results = []
    for sentence in sentences:
        cleaned = clean_text(sentence)
        vec = vectorizer.transform([cleaned])
        prob = model.predict_proba(vec)[0][1]
        label = "THREAT" if prob >= threshold else "SAFE"
        results.append({
            "sentence":sentence,
            "label":label,
            "prob": round(prob * 100,1)
        })
    # overall verdict — threat if ANY sentence triggered
    overall = "THREAT" if any(r["label"] == "THREAT" for r in results) else "SAFE"
    
    # highest threat probability across all sentences
    max_prob = max(r["prob"] for r in results)
    
    return overall, max_prob, results