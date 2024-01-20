from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from joblib import dump, load
from sklearn.metrics import accuracy_score
import pandas as pd
import os
from html.parser import HTMLParser
import email
import string
import nltk

class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

class Parser:
    def __init__(self):
        self.stemmer = nltk.PorterStemmer()
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self.punctuation = list(string.punctuation)
        
    def parse(self, email_path):
        with open(email_path, errors='ignore') as e:
            msg = email.message_from_file(e)
        return None if not msg else self.get_email_content(msg)

    def get_email_content(self, msg):
        subject = self.tokenize(msg['Subject']) if msg['Subject'] else []
        body = self.get_email_body(msg.get_payload(), msg.get_content_type())
        content_type = msg.get_content_type()
        return {"subject": subject, "body": body, "content_type": content_type}
                
    def get_email_body(self, payload, content_type):
        body = []
        if type(payload) is str and content_type == 'text/plain':
            return self.tokenize(payload)
        elif type(payload) is str and content_type == 'text/html':
            return self.tokenize(strip_tags(payload))
        elif type(payload) is list:
            for p in payload:
                body += self.get_email_body(p.get_payload(), p.get_content_type())
        return body
        
    def tokenize(self, text):
        for c in self.punctuation:
            text = text.replace(c, "")
        text = text.replace("\t", " ")
        text = text.replace("\n", " ")
        tokens = list(filter(None, text.split(" ")))
        return [self.stemmer.stem(w) for w in tokens if w not in self.stopwords]

def parse_index(path_to_index, n_elements, dataset_path):
    ret_indexes = []
    index = open(path_to_index).readlines()
    for i in range(n_elements):
        mail = index[i].split(" ../")
        label = mail[0]
        path = mail[1][:-1]
        ret_indexes.append({"label": label, "email_path": os.path.join(dataset_path, path)})
    return ret_indexes

def parse_email(index):
    p = Parser()
    email_path = "../" + index["email_path"]
    pmail, label = p.parse(email_path), index["label"]
    print(f"Parsing email: {email_path}, Label: {label}")
    return pmail, label

def create_prep_dataset(index_path, n_elements, dataset_path):
    X = []
    y = []
    indexes = parse_index(index_path, n_elements, dataset_path)
    for i in range(n_elements):
        print("\rParsing email: {0}".format(i+1), end='')
        mail, label = parse_email(indexes[i])
        X.append(" ".join(mail['subject']) + " ".join(mail['body']))
        y.append(label)
    return X, y

def train_model(X_train, y_train):
    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    
    model = LogisticRegression()
    model.fit(X_train_vectorized, y_train)
    
    dump((vectorizer, model), 'modelo_entrenado.joblib')
    
    return vectorizer, model

def evaluate_model(model, vectorizer, X_test, y_test):
    X_test_vectorized = vectorizer.transform(X_test)
    y_predict = model.predict(X_test_vectorized)
    
    accuracy = accuracy_score(y_test, y_predict)
    return accuracy

def main():
    DATASET_PATH = "datasets/trec07p"
    X_train, y_train = create_prep_dataset("../datasets/trec07p/full/index", 100, DATASET_PATH)
    
    vectorizer, model = train_model(X_train, y_train)
    
    X_test, y_test = create_prep_dataset("../datasets/trec07p/full/index", 2000, DATASET_PATH)


    # Asegúrate de que X_test y y_test tengan la misma longitud
    if len(X_test) != len(y_test):
        raise ValueError("Las longitudes de X_test e y_test son diferentes")

    accuracy = evaluate_model(model, vectorizer, X_test, y_test)
    print('Accuracy: {:.3f}'.format(accuracy))

if __name__ == "__main__":
    main()
