from nltk.stem import PorterStemmer
import string

def tokenization(query) :
    with open("data/stopwords.txt", 'r', encoding="utf-8") as file:
        stopwords = file.read()
    stopwords = stopwords.splitlines()

    tranDict = {}
    for punc in string.punctuation:
        tranDict[punc] = ""
    tranTable = str.maketrans(tranDict)

    queryItems = query.lower().translate(tranTable).split()
    queryItems = [x for x in queryItems if x.strip() != ""]
    queryItems = [x for x in queryItems if x not in stopwords]

    stemmer = PorterStemmer()
    queryItems = [stemmer.stem(x) for x in queryItems]

    return queryItems