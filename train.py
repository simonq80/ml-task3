from sklearn import svm, metrics, preprocessing
import csv
#import _pickle
import random
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.utils import shuffle
from sklearn import neighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np


def dataset_from_csv(datafile, independent_columns, color_columns, text_columns, dependent_column, valid_classes):
    independent_vars = []
    dependent_vars = []

    with open(datafile) as csvfile:
            datagen = csv.reader(csvfile, delimiter=',')
            column_heads = next(datagen)
            # Get Indexes of used columns
            independent_indexes = [column_heads.index(x) for x in independent_columns]
            color_indexes = [column_heads.index(x) for x in color_columns]
            text_indexes = [column_heads.index(x) for x in text_columns]
            dependent_index = column_heads.index(dependent_column)

            for row in datagen:
                if row[dependent_index] in valid_classes: #Only take known genders
                    # split colors into seperate rgb values
                    rgbs = []
                    for x in color_indexes:
                        try:
                            r = int(row[x][:2], 16)
                            g = int(row[x][2:4], 16)
                            b = int(row[x][4:], 16)
                            rgbs.append(r)
                            rgbs.append(g)
                            rgbs.append(b)
                        except ValueError:
                            rgbs += [0,0,0]

                    independent_vars.append(([row[x] for x in text_indexes] + [int(row[x]) for x in independent_indexes] + rgbs))
                    dependent_vars.append(row[dependent_index])
    return independent_vars, dependent_vars



datafile = 'twitter_gender.csv'
independent_columns = ['fav_number', 'tweet_count']
color_columns = ['link_color', 'sidebar_color']
text_columns = ['description']
dependent_column = 'gender'
valid_classes = ['male', 'female', 'brand']

independent_vars, dependent_vars = dataset_from_csv(datafile, independent_columns, color_columns, text_columns, dependent_column, valid_classes)


### Text Processing
def train_text_cls(dataset, target, numrows):
    text_classifiers = []
    for i in range(0, numrows): #Create text classifier for each text column
        mnb_clf = Pipeline([('vect', CountVectorizer()),
                            ('tfidf', TfidfTransformer()),
                            ('clf',  MultinomialNB()),
                            ])
        mnb_parameters = {'vect__ngram_range': [(1, 2)],
            'tfidf__use_idf': ([True]),
            'clf__alpha': ([0.75]),
            'clf__fit_prior': ([True]),
            }

        gs_clf = GridSearchCV(mnb_clf, mnb_parameters, n_jobs=-1,cv=3)
        gs_clf.fit([x[i] for x in dataset], target)
        text_classifiers.append(gs_clf)
    return text_classifiers


def process_text(dataset, clfs, numrows):

    for row in dataset:
        for clf in range(0, numrows):
            pred = clfs[clf].predict([row[clf]])[0]
            new_vals = [0,0,0]
            new_vals[valid_classes.index(pred)] = 1
            row += new_vals

    dataset = preprocessing.scale([x[numrows:] for x in dataset])

    return dataset

def split_test(indep_vars, dep_vars, test_size=0.3, offset=0):
    start = int(len(dep_vars) * offset)
    stop = int(len(indep_vars) * (offset + test_size))
    x_train = indep_vars[:start] + indep_vars[stop:]
    x_test = indep_vars[start:stop]
    y_train = dep_vars[:start] + dep_vars[stop:]
    y_test = dep_vars[start:stop]

    clfs = train_text_cls(x_train, y_train, len(text_columns))
    x_train = process_text(x_train, clfs, len(text_columns))
    x_test = process_text(x_test, clfs, len(text_columns))

    c = svm.SVC(probability=False)
    c.fit(x_train, y_train)
    return c.score(x_test, y_test)

def nfold(indep_vars, dep_vars, folds=5):
    results = []
    for i in range(0, folds):
        results.append(split_test(indep_vars, dep_vars, 1.0/folds, 1.0/folds*i))

    return results




if __name__ == "__main__":
    print('70:30 Split:')
    print(split_test(independent_vars, dependent_vars))
    print('5-Fold:')
    nf = nfold(independent_vars, dependent_vars)
    print(nf)
    print('Mean:')
    print(np.mean(nf))



    #with open('svm.pkl', 'wb') as fid: #save svm
    #    _pickle.dump(c, fid)
