from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def clean_text(text):
  words = text.lower().split()
  return " ".join([w for w in words if w not in ENGLISH_STOP_WORDS and len(w) > 2])
