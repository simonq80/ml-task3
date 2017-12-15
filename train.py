from sklearn import svm, metrics, preprocessing
import csv
import _pickle
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


x_train, x_test, y_train, y_test = train_test_split(independent_vars, dependent_vars, test_size=0.3)


### Text Processing
print('Processing Text')
text_classifiers = []
for i in range(0, len(text_columns)): #Create text classifier for each text column
    mnb_clf = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('clf',  MultinomialNB()),
                        ])
    mnb_parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
        'tfidf__use_idf': (True, False),
        'clf__alpha': (0.1,0.25,0.5,0.75, 1.0),
        'clf__fit_prior': (True, False),
        }

    gs_clf = GridSearchCV(mnb_clf, mnb_parameters, n_jobs=-1,cv=3)
    gs_clf.fit([x[i] for x in x_train], y_train)
    text_classifiers.append(gs_clf)

for row in x_train:
    for clf in range(0, len(text_columns)):
        pred = text_classifiers[clf].predict([row[clf]])[0]
        new_vals = [0,0,0]
        new_vals[valid_classes.index(pred)] = 1
        row += new_vals

x_train = preprocessing.scale(x_train)

print(x_train[:20])


if __name__ == "__main__":
    c = svm.SVC(probability=False)
    print('Training SVM...')
    c.fit(x_train, y_train)
    print(c.score(x_test, y_test))

    c = svm.SVC()
    print(cross_val_score(c, independent_vars, dependent_vars, cv=5))



    with open('svm.pkl', 'wb') as fid: #save svm
        _pickle.dump(c, fid)
