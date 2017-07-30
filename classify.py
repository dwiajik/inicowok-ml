import argparse, pickle

def predict(name, classifier):
    with open('./pickles/{}.pkl'.format(classifier), 'rb') as dump_file:
        model = pickle.load(dump_file)
    
    return model.predict([name.lower()])[0]

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

    result = predict(args.name, args.ml or 'nb')
    print ('Gender prediction using {} classifier:'.format(ml_type))
    label = {1: 'Pria', 0: 'Wanita'}
    print(args.name, ': ', label[result])

# args setting
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description = 'Decide gender from someone\'s name')
 
  parser.add_argument(
                      'name',
                      help = 'Name',
                      metavar='name'
                      )
  parser.add_argument(
                      '-ml',
                      help = 'nb=Naive Bayes(default); lg=Logistic Regression; rf=Random Forest; svm=Support Vector Machine',
                      choices=['nb', 'lg', 'rf', 'svm']
                      )
  args = parser.parse_args()
  
  main(args)