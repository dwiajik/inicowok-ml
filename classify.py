import sys, argparse, pickle, os
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

# load dataset
def load_data(dataset):
    df = pd.read_csv(dataset, encoding='utf-8-sig')
    df = df.dropna(how='all')
    
    jk_map = {'male' : 1, 'female' : 0}
    df['gender'] = df['gender'].map(jk_map)

    feature_col_names = ['name']
    predicted_class_names = ['gender']
    X = df[feature_col_names].values     
    y = df[predicted_class_names].values 
    
    return (X, y)

def predict(name, dataset, classifier):
    cv = StratifiedKFold(n_splits=10, shuffle=True)

    classifiers = {
        'nb': MultinomialNB(),
        'lg': LogisticRegression(),
        'rf': RandomForestClassifier(n_estimators=10, n_jobs=-1),
        'svm': LinearSVC(),
    }

    if os.path.isfile('./pickles/{}.pkl'.format(classifier)) and dataset is None:        
        dump_file = open('./pickles/{}.pkl'.format(classifier), 'rb')
        model = pickle.load(dump_file)
    else:
        dump_file = open('./pickles/{}.pkl'.format(classifier), 'wb')
        pipeline = Pipeline([
            ('vect', CountVectorizer(analyzer = 'char_wb', ngram_range=(2,6))),
            ('tfidf', TfidfTransformer()),
            ('clf', classifiers[classifier])
        ])

        #train and dump to file                     
        dataset = load_data(dataset or './corpora/names.csv')
        model = pipeline.fit(dataset[0].ravel(), dataset[1].ravel())
        pickle.dump(model, dump_file)
        
        #Akurasi
        scores = cross_val_score(model, dataset[0].ravel(), dataset[1].ravel(), cv=cv)
        accuracy = scores.mean()
        print('Akurasi: {}%'.format(accuracy))
    
    return model.predict([name])[0]

# main
def main(args):
    if(args.ml == 'lg'):
        ml_type = 'Logistic Regression'
    elif(args.ml == 'rf'):
        ml_type = 'Random Forest'
    elif(args.ml == 'svm'):
        ml_type = 'Support Vector Machine'
    else:
        ml_type = 'Naive Bayes'
    
    result = predict(args.name, args.train, args.ml or 'nb')
    print ('Prediksi jenis kelamin dengan', ml_type, ':')
    jk_label = {1:'Pria', 0:'Wanita'}
    print(args.name, ': ', jk_label[result])

# args setting
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description = 'Menentukan jenis kelamin berdasarkan nama Bahasa Indoensia')
 
  parser.add_argument(
                      'name',
                      help = 'Nama',
                      metavar='nama'
                      )
  parser.add_argument(
                      '-ml',
                      help = 'NB=Naive Bayes(default); LG=Logistic Regression; RF=Random Forest',
                      choices=['nb', 'lg', 'rf', 'svm']
                      )
  parser.add_argument(
                      '-t',
                      '--train',
                      help='Training ulang dengan dataset yang ditentukan')
  args = parser.parse_args()
  
  main(args)