from sklearn import svm, metrics, preprocessing
import csv
import _pickle
import random
from sklearn.model_selection import train_test_split, cross_val_score
import colorsys


def dataset_from_csv(datafile, independent_columns, color_columns, dependent_column, valid_classes):
    independent_vars = []
    dependent_vars = []

    with open(datafile) as csvfile:
            datagen = csv.reader(csvfile, delimiter=',')
            column_heads = next(datagen)
            # Get Indexes of used columns
            independent_indexes = [column_heads.index(x) for x in independent_columns]
            color_indexes = [column_heads.index(x) for x in color_columns]
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

                    independent_vars.append(([int(row[x]) for x in independent_indexes] + rgbs))
                    dependent_vars.append(row[dependent_index])
    return independent_vars, dependent_vars



datafile = 'twitter_gender.csv'
independent_columns = ['fav_number', 'tweet_count']
color_columns = ['link_color', 'sidebar_color']
dependent_column = 'gender'
valid_classes = ['male', 'female', 'brand']

independent_vars, dependent_vars = dataset_from_csv(datafile, independent_columns, color_columns, dependent_column, valid_classes)
independent_vars = preprocessing.scale(independent_vars)

x_train, x_test, y_train, y_test = train_test_split(independent_vars, dependent_vars, test_size=0.3)

if __name__ == "__main__":
    c = svm.SVC(probability=False)
    print('Training SVM...')
    c.fit(x_train, y_train)
    print(c.score(x_test, y_test))

    c = svm.SVC()
    print(cross_val_score(c, independent_vars, dependent_vars, cv=5))



    with open('svm.pkl', 'wb') as fid: #save svm
        _pickle.dump(c, fid)
