# data preprocess package
import pandas as pd

# text file related os module
from os import listdir
from os.path import isfile, join

# necessary module for clean and process file data content
from nltk.corpus import stopwords
import string

# necessary scikit learn module and algorithm for build pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import Perceptron
from sklearn.pipeline import Pipeline

import warnings
import pickle
import nltk

warnings.filterwarnings("ignore")
nltk.download('stopwords')

# file type
file_type = ["cantera", "chemkin"]

# read train file data content
train_file_content = []
train_file_content_type = []

for types in file_type:
    train_path = "training data/{}/".format(types)
    for file in listdir(train_path):
        if isfile(join(train_path, file)):
            file = open("training data/{0}/{1}".format(types, file), encoding="utf8", errors='ignore')
            content = file.read()
            train_file_content.append(content)
            train_file_content_type.append(types)

# read test file data content
test_file_content = []
test_file_content_type = []

for types in file_type:
    test_path = "test data/{}/".format(types)
    for file in listdir(test_path):
        if isfile(join(test_path, file)):
            file = open("test data/{0}/{1}".format(types, file), encoding="utf8", errors='ignore')
            content = file.read()
            test_file_content.append(content)
            test_file_content_type.append(types)

# dataframe of train file data content
train_data = {'File Data': train_file_content,
              'File Type': train_file_content_type}

train_df = pd.DataFrame(train_data)

# dataframe of test file data content
test_data = {'File Data': test_file_content,
             'File Type': test_file_content_type}

test_df = pd.DataFrame(test_data)


def file_content_process(content):
    """
        1. remove punctuation from file content
        2. remove stopwords from file content
        3. return list of clean file content
    """
    nopunc_content = [c for c in content if c not in string.punctuation]
    nopunc_content = ''.join(nopunc_content)

    return [word for word in nopunc_content.split() if word.lower() not in stopwords.words('english')]


pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=file_content_process)),  # this work is convert file data to vectorizing wordwise
    ('tfidf', TfidfTransformer()),  # this issue is to normalize the vector with the respective magnitudes
    ('model', Perceptron())  # ann classification scikit learn algorithm
])

# fit training data in pipeline
pipeline.fit(train_df['File Data'], train_df['File Type'])

pickle.dump(pipeline, open('ann_nlp_model.pkl', 'wb'))
model = pickle.load(open('ann_nlp_model.pkl', 'rb'))
print(model.predict(test_df['File Data']))
