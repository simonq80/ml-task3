import numpy as np
import pandas as pd
from pathlib import Path
import zipfile
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


dataP2 = "twitter_gender.csv"


# play with dataset dataP2

methodSize = [1]
myfile = Path(dataP2)
if not myfile.is_file():
    zip_ref = zipfile.ZipFile('twitter-user-gender-classification'+'.zip','r')
    zip_ref.extractall()
    zip_ref.close()
# load data, shuffle,
dataset = pd.read_csv(dataP2,sep=',',encoding="latin-1")
dataset=dataset.sample(random_state =1,frac=1)

# replace NaN in gender column with 'unkown'
# then only work with the rows with gender = 'male' or 'female'
dataset['gender'].replace(np.nan,'unknown',inplace=True)
dataset = dataset[dataset.gender!='unknown']
xData =dataset['description'].values.astype('U')
yTarget = dataset['gender'].values.astype('U')

#tfidf_transformer = TfidfTransformer()
#count_vect = CountVectorizer()
#X_counts = count_vect.fit_transform(xData)
#X_tfidf = tfidf_transformer.fit_transform(X_counts)

# split into trainset and testset, 0.8 and 0.2
X,X_test,Y,Y_test = train_test_split(xData,yTarget,test_size=0.3,random_state=0,shuffle=True)

X = np.array(X)
Y = np.array(Y)

#   do count and tfidf in pipeline with a selected classifier
#   count -> tfidf -> classifier

#   Multinomial Naive Bayes classifier

mnb_clf = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf',  MultinomialNB()),
                    ])
mnb_parameters = {
    'clf__alpha': [(1.0)],
}
# K = 5
gs_clf = GridSearchCV(mnb_clf, mnb_parameters, n_jobs=-1,cv=5)
gs_clf.fit(X,Y)
print()
print('GridSearchCV using mnb_clf&mnb_parameters. Here are the best score and best params')
print(gs_clf.best_score_)
print(gs_clf.best_params_)
print()
means = gs_clf.cv_results_['mean_test_score']
stds = gs_clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, gs_clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()
print('score of test set:')
# 3 fold cross validation on the classifier which has best parameters
scores = cross_val_score(gs_clf,X_test,Y_test,cv =3)
print(np.mean(scores))

print()

#   SVM Linear
svm_clf = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf',  SGDClassifier()),
                    ])
svm_parameters = {'vect__ngram_range': [(1, 2)],
    'tfidf__use_idf': [(True)],
    'clf__alpha': (1.0, 0.1, 1e-2, 1e-3, 1e-4, 1e-5),
}
gs_clf = GridSearchCV(svm_clf, svm_parameters, n_jobs=-1,cv=5)
gs_clf.fit(X,Y)
print('GridSearchCV using sgdc_clf&sgdc_parameters. Here are the best score and best params')
print(gs_clf.best_score_)
print(gs_clf.best_params_)
print()
means = gs_clf.cv_results_['mean_test_score']
stds = gs_clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, gs_clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()
print('score of test set:')
scores = cross_val_score(gs_clf,X_test,Y_test,cv =3)
print(np.mean(scores))
print()

# mlp
mlp_clf = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MLPClassifier(alpha=1e-05, hidden_layer_sizes=(5,), learning_rate='constant',
                                          learning_rate_init=0.001, shuffle=True,
                                          solver='lbfgs')),
                    ])
mlp_parameters = {
    'clf__alpha': (1.0, 0.1, 1e-2, 1e-3, 1e-4, 1e-5),
}

gs_clf = GridSearchCV(mlp_clf, mlp_parameters, n_jobs=-1,cv=5)
gs_clf.fit(X,Y)
print('GridSearchCV using mlp_clf&mlp_parameters. Here are the best score and best params')
print(gs_clf.best_score_)
print(gs_clf.best_params_)
print()
means = gs_clf.cv_results_['mean_test_score']
stds = gs_clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, gs_clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()
print('score of test set:')
scores = cross_val_score(gs_clf,X_test,Y_test,cv =3)
print(np.mean(scores))
print()
