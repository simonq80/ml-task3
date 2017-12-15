# Machine Learning - Assignment 3

## 1. Data & Pre-processing

The task was to develop a system that predicts the gender of a Twitter user. The dataset contains many instances where the gender feature was labelled as "unknown" or had no label at all. When training our model, we wanted to avoid teaching the model to identify users with an unknown gender. Therefore, the instances with no gender label or with an "unknown" label were removed from the dataset.

## 2. Algorithm & Feature Selection

Our system is divided into 2 main sections - text processing and number processing.

### 2a. Text Processing

In text processing, we investigated if the profile's description could be used as an indication of the profile user's gender. We tested three algorithms, Multinomial Naïve Bayes Classifier (MNB), Stochastic Gradient Descent Classifier (SGDC), Multi-layer Perceptron Classifier (MLP).

We used the "description" feature as our variable and "gender" as our target. The text of the descriptions was tokenized and a dictionary was built which mapped the occurances of different words to the gender. This was done through Scikit's Count Vectorizer. We took this approach because attempting to classify the gender according to whole texts or sentences is inaccurate and ineffecient. Through tokenization, we were able to reframe the problem from a text classification to a number classification. Finally, we used TF-IDF to convert term occurances into term frequencies accosiated with the genders.

Next was choosing the best parameters. We ran a grid search cross validation on all three algorithms to find the parameters that gave the best accuracy score for each algorithm. The parameters that produced that highest accuracy are highlighted in bold in their tables.

#### Multinomial Naïve Bayes Classifier parameters

| Accuracy | Alpha | Fit Prior | Use TF-IDF | N-gram Range |
| --- | --- | --- | --- | --- |
| 0.594 | 0.10 | TRUE  | TRUE  | (1, 1) |
| 0.600 | 0.10 | TRUE  | TRUE  | (1, 2) |
| 0.604 | 0.10 | TRUE  | FALSE | (1, 1) |
| 0.606 | 0.10 | TRUE  | FALSE | (1, 2) |
| 0.585 | 0.10 | FALSE | TRUE  | (1, 1) |
| 0.590 | 0.10 | FALSE | TRUE  | (1, 2) |
| 0.596 | 0.10 | FALSE | FALSE | (1, 1) |
| 0.596 | 0.10 | FALSE | FALSE | (1, 2) |
| 0.601 | 0.25 | TRUE  | TRUE  | (1, 1) |
| 0.605 | 0.25 | TRUE  | TRUE  | (1, 2) |
| 0.608 | 0.25 | TRUE  | FALSE | (1, 1) |
| 0.608 | 0.25 | TRUE  | FALSE | (1, 2) |
| 0.594 | 0.25 | FALSE | TRUE  | (1, 1) |
| 0.596 | 0.25 | FALSE | TRUE  | (1, 2) |
| 0.598 | 0.25 | FALSE | FALSE | (1, 1) |
| 0.599 | 0.25 | FALSE | FALSE | (1, 2) |
| 0.606 | 0.50 | TRUE  | TRUE  | (1, 1) |
| 0.609 | 0.50 | TRUE  | TRUE  | (1, 2) |
| 0.608 | 0.50 | TRUE  | FALSE | (1, 1) |
| 0.609 | 0.50 | TRUE  | FALSE | (1, 2) |
| 0.599 | 0.50 | FALSE | TRUE  | (1, 1) |
| 0.599 | 0.50 | FALSE | TRUE  | (1, 2) |
| 0.600 | 0.50 | FALSE | FALSE | (1, 1) |
| 0.601 | 0.50 | FALSE | FALSE | (1, 2) |
| 0.608 | 0.75 | TRUE  | TRUE  | (1, 1) |
| **0.609** | **0.75** | **TRUE**  | **TRUE**  | **(1, 2)** |
| 0.606 | 0.75 | TRUE  | FALSE | (1, 1) |
| 0.608 | 0.75 | TRUE  | FALSE | (1, 2) |
| 0.600 | 0.75 | FALSE | TRUE  | (1, 1) |
| 0.601 | 0.75 | FALSE | TRUE  | (1, 2) |
| 0.599 | 0.75 | FALSE | FALSE | (1, 1) |
| 0.599 | 0.75 | FALSE | FALSE | (1, 2) |
| 0.608 | 1.00 | TRUE  | TRUE  | (1, 1) |
| 0.607 | 1.00 | TRUE  | TRUE  | (1, 2) |
| 0.605 | 1.00 | TRUE  | FALSE | (1, 1) |
| 0.606 | 1.00 | TRUE  | FALSE | (1, 2) |
| 0.600 | 1.00 | FALSE | TRUE  | (1, 1) |
| 0.602 | 1.00 | FALSE | TRUE  | (1, 2) |
| 0.597 | 1.00 | FALSE | FALSE | (1, 1) |
| 0.599 | 1.00 | FALSE | FALSE | (1, 2) |

#### Stochastic Gradient Descent Classifier parameters

| Accuracy | Alpha | Use TF-IDF | N-gram Range |
| --- | --- | --- | --- |
| 0.587 | 0.001  | TRUE  | (1, 1) |
| 0.586 | 0.001  | TRUE  | (1, 2) |
| 0.569 | 0.001  | FALSE | (1, 1) |
| 0.571 | 0.001  | FALSE | (1, 2) |
| 0.602 | 0.0001 | TRUE  | (1, 1) |
| **0.603** | **0.0001** | **TRUE**  | **(1, 2)** |
| 0.597 | 0.0001 | FALSE | (1, 1) |
| 0.600 | 0.0001 | FALSE | (1, 2) |

#### Multi-layer Perceptron Classifier parameters

| Accuracy | Alpha | Use TF-IDF |
| 0.527 | 0.0001   | TRUE  |
| 0.517 | 0.0001   | FALSE |
| **0.567** | **0.00001**  | **TRUE**  |
| 0.561 | 1.00E-05 | FALSE |

### 2b. Number Processing

In number processing, we used the SVM algorithm using the fav_number, tweet_count, link_color and sidebar_color as our independent features. Of course, gender was our target feature. The gender feature was converted to dummy variables using One Hot Enoding. The link_color and sidebar_color features are hex values and so, for number processing, we converted the hex value into three features representing the red, green and blue values of the colours. Finally, the features used were normalised. These edited features were then used to train the model and make predictions.

TODO : Finding best parameters

### 2c. Combining Text and Number Processing

## 3. Evaluation

### 3a. Text Processing

From Section 2a, we have the best parameters for each of the three text processing algorithms. We compared these three algorithms against each other to identify which one provides the most accurate results. We performed a 3-fold cross validation. We found that MNB classifier is the most accurate. However, the accuracy is only 0.565. Although it is better than guessing the gender randomly (33% chance of being correct), it is not reliable enough for a prediction system.

### 3b. Number Processing

TODO: Talk about accuracy 

### 3c. Text and Number Processing

TODO : Talk about accuracy

#### Algorithm accuracies

| MNB | SGDC | MLP |
| --- | --- | --- |
| 0.565 | 0.525 | 0.506 |

## 4. Conclusion

## 5. Appendix

| Name | Student # | Contributions |
| --- | --- | --- |
| Jinwei Yuan | 17306137 | Main developer of the text processing code - developed multinominal naive bayes, linear SVM and MLP classifiers |
| Simon Quigley | id | Main developer of the number processing code - developed *** |
| Patrick Geoghegan | 13320590 | Assited in development, analysis of results and writer of report |

## 6. Source Code

```python
print("Code goes here")
```
