import re
def clean_text(message):
    cleanedmessage = message.lower()
    cleanedmessage = re.sub(r'http\S+', 'url', cleanedmessage)
    cleanedmessage = re.sub(r'\d+','num', cleanedmessage)
    cleanedmessage = re.sub(r'[^a-z\s]', '',cleanedmessage)
    cleanedmessage = re.sub(r'\s+',' ', cleanedmessage)
    return cleanedmessage
