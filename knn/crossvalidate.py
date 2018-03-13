import numpy as np

from src.algorithms.knn import KnnClassifier


def generate_partition(full_data, subset_count):
    # p-fold cross-validation
    df = full_data.copy()
    partition_count = len(df) // subset_count
    subsets = []
    for i in range(subset_count - 1):
        subset = df.sample(partition_count, random_state=1337)
        df = df.drop(subset.index)
        subsets.append(subset)
    subsets.append(df)
    return subsets


def cross_validate(k, metric, full_data, subsets):
    results = []
    for i in range(len(subsets)):
        test_data = subsets[i]
        train_data = full_data.drop(test_data.index)
        classifier = KnnClassifier(k, train_data, metric)
        test_result = test_classification(classifier, test_data)
        results.append(test_result)
    return sum(results) / len(results)


def test_classification(classifier, test_data):
    correct_count = 0
    for _, row in test_data.iterrows():
        label = classifier.classify(np.array(row[:-1]))
        if label == row['cls']:
            correct_count += 1
    return correct_count / len(test_data)
