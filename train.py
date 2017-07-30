import argparse, pickle
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import pandas as pd

# load dataset
def load_data(dataset):
    df = pd.read_csv(dataset, encoding='utf-8-sig')
    df = df.dropna(how='all')
    
    jk_map = {'male' : 1, 'female' : 0}
    df['gender'] = df['gender'].map(jk_map)

    X = df['name'].values
    y = df['gender'].values
    
    return (X, y)

def train(dataset, classifier):
    cv = StratifiedKFold(n_splits=10, shuffle=True)
    classifiers = {
        'nb': MultinomialNB(),
        'lg': LogisticRegression(),
        'rf': RandomForestClassifier(n_estimators=10, n_jobs=-1),
        'svm': LinearSVC(),
    }

    pipeline = Pipeline([
        ('vect', CountVectorizer(analyzer = 'char_wb', ngram_range=(2,6))),
        ('tfidf', TfidfTransformer()),
        ('clf', classifiers[classifier])
    ])

    #train and dump to file                     
    dataset = load_data(dataset or './corpora/names.csv')
    model = pipeline.fit([x.lower() for x in dataset[0].ravel()], dataset[1].ravel())

    with open('./pickles/{}.pkl'.format(classifier), 'wb') as dump_file:
        pickle.dump(model, dump_file)
    
    #Akurasi
    scores = cross_val_score(pipeline, [x.lower() for x in dataset[0].ravel()], dataset[1].ravel(), cv=cv)
    accuracy = scores.mean()
    print('Accuracy: {}%'.format(round(accuracy * 100, 2)))

    return model


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

    train(args.dataset, args.ml or 'nb')
    print('Successfully train {} classifier!'.format(ml_type))

# args setting
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description = 'Train name gender classifier')

  parser.add_argument(
                      'dataset',
                      help = 'Dataset',
                      metavar='dataset'
                      )
  parser.add_argument(
                      '-ml',
                      help = 'nb=Naive Bayes(default); lg=Logistic Regression; rf=Random Forest; svm=Support Vector Machine',
                      choices=['nb', 'lg', 'rf', 'svm']
                      )
  args = parser.parse_args()
  
  main(args)