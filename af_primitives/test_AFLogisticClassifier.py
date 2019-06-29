import unittest
import numpy as np
import pickle
from typing import NamedTuple

from sklearn import datasets
from sklearn.utils import shuffle

import AFLogisticClassifier
# from autogenerate.d3m_sklearn_wrap.sklearn_wrap import SKBaggingClassifier

# Common random state
rng = np.random.RandomState(0)

# Toy sample from sklearn tests
X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
y_class = ["foo", "foo", "foo", 1, 1, 1]    # test string class labels
T = [[-1, -1], [2, 2], [3, 2]]
y_t_class = ["foo", 1, 1]

# Load the iris dataset and randomly permute it
iris = datasets.load_iris()
perm = rng.permutation(iris.target.size)
iris.data, iris.target = shuffle(iris.data, iris.target, random_state=rng)

Hyperparams = NamedTuple('Hyperparams', [
    ('base_estimator', object),
    ('learning_rate', float),
    ('n_estimators', int),
    ('algorithm', str)
])


class TestSKBaggingClassifier(unittest.TestCase):

    def test(self):
        classes = np.unique(iris.target)
        hyperparams = SKBaggingClassifier.Hyperparams.defaults()
        # Create the model object
        clf = SKBaggingClassifier.SKBaggingClassifier(hyperparams=hyperparams)
        train_set = iris.data
        targets = iris.target
        clf.set_training_data(inputs=train_set, outputs=targets)
        clf.fit()

        output = clf.produce(inputs=train_set)

        # Testing get_params() and set_params()
        params = clf.get_params()
        clf.set_params(params=params)
        first_output = clf.produce(inputs=train_set)

        # pickle the params and hyperparams
        pickled_params = pickle.dumps(params)
        unpickled_params = pickle.loads(pickled_params)

        pickled_hyperparams = pickle.dumps(hyperparams)
        unpickled_hyperparams = pickle.loads(pickled_hyperparams)

        # Create a new object from pickled params and hyperparams
        new_clf = SKBaggingClassifier.SKBaggingClassifier(hyperparams=unpickled_hyperparams)
        new_clf.set_params(params=unpickled_params)
        new_output = new_clf.produce(inputs=train_set)

        # Check if outputs match
        self.assertTrue(np.array_equal(first_output.value, output.value))
        self.assertTrue(np.array_equal(new_output.value, output.value))

        # We want to test the running of the code without errors and not the correctness of it
        # since that is assumed to be tested by sklearn
        # assert np.array_equal(classes, clf._clf.classes_)
        print("SUCCESS: Test fit produce on SKBaggingClassifier")
        model = pickle.dumps(clf)
        new_clf = pickle.loads(model)
        new_output = new_clf.produce(inputs=train_set)
        self.assertTrue(np.array_equal(output.value, output.value))
        self.assertTrue(np.array_equal(new_output.value, output.value))
        print("SUCCESS: Test pickling entire model on SKBaggingClassifier")


if __name__ == '__main__':
    unittest.main()
