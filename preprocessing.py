import re
 
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess_comment(comment):
    comment = comment.lower().strip()
    # pungtuation
    comment = ' '.join([re.sub("[^A-Za-z0-9\s?!.,|*]",' ',word) for word in comment.split()])
    # stopwords
    STOPWORDS = set(stopwords.words("english")) -{'not','no','yes','yet','however','but'}  # exception this stopwords--{'not','no','yes','yet','however','but'}
    comment = ' '.join([word for word in comment.split() if word not in STOPWORDS])
    # Lemmatizer
    lemmatizer = WordNetLemmatizer()
    comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

    return comment

LABEL = {0: "negative", 1: "neutral", 2: "positive"}