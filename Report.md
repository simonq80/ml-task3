# Machine Learning - Assignment 3

## Data & Pre-processing

The task was to develop a system that predicts the gender of a Twitter user. The dataset contains many instances where the gender feature was labelled as "unknown" or had no label at all. When training our model, we wanted to avoid teaching the model to identify users with an unknown gender. Therefore, the instances with no gender label or with an "unknown" label were removed from the dataset.

## Algorithm & Feature Selection

Our system is divided into 2 main sections - text processing and number processing.

In text processing, we investigated if the profile's description could be used as an indication of the profile user's gender. We tested three algorithms, multinominal naive bayes classifier, linear SVM classifier and MLP classifier.

[//]: # (TODO: List the benefits and downsides of each algorithm)

## Evaluation

## Conclusion

## Appendix

| Name | Student # | Contributions |
| --- | --- | --- |
| Jinwei Yuan | 17306137 | Main developer of the text processing code - developed multinominal naive bayes, linear SVM and MLP classifiers |
| Simon Quigley | id | Main developer of the number processing code - developed *** |
| Patrick Geoghegan | 13320590 | Assited in development, analysis of results and writer of report |

## Source Code

```python
print("Code goes here")
```
