import unittest
import numpy as np
import pickle
from typing import NamedTuple

from sklearn import datasets
from sklearn.utils import shuffle

import AFLogisticRegression
# from autogenerate.d3m_sklearn_wrap.sklearn_wrap import AFLogisticRegression

# Taken from original af logit example
import arrayfire as af
from arrayfire.algorithm import count, imax, sum
from arrayfire.arith import abs, log, sigmoid
from arrayfire.blas import matmul, matmulTN
from arrayfire.data import constant, join
from arrayfire.device import eval, sync
from arrayfire.interop import from_ndarray

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
    ('learning_rate', float),
    ('reg_constant', float),
    ('max_err', float),
    ('max_iter', int),
    ('verbose', int),
])


def ints_to_onehots(ints, num_classes):
    onehots = np.zeros((ints.shape[0], num_classes), dtype='float32')
    onehots[np.arange(ints.shape[0]), ints] = 1
    return onehots


def onehots_to_ints(onehots):
    return np.argmax(onehots, axis=1)


def accuracy(predicted, target):
    _, tlabels = af.imax(target, 1)
    _, plabels = af.imax(predicted, 1)
    return 100 * af.count(plabels == tlabels) / tlabels.elements()


def abserr(predicted, target):
    return 100 * af.sum(af.abs(predicted - target)) / predicted.elements()


class RefAfLogisticRegression:
    def __init__(self, alpha=0.1, lambda_param=1.0, maxerr=0.01, maxiter=1000, verbose=False):
        self.__alpha = alpha
        self.__lambda_param = lambda_param
        self.__maxerr = maxerr
        self.__maxiter = maxiter
        self.__verbose = verbose
        self.__weights = None


    def predict_proba(self, X):
        Z = af.matmul(X, self.__weights)
        return af.sigmoid(Z)


    def predict_log_proba(self, X):
        return af.log(self.predict_proba(X))


    def predict(self, X):
        probs = self.predict_proba(X)
        _, classes = af.imax(probs, 1)
        return classes


    def cost(self, X, Y):
        # Number of samples
        m = Y.dims()[0]

        dim0 = self.__weights.dims()[0]
        dim1 = self.__weights.dims()[1] if len(self.__weights.dims()) > 1 else None
        dim2 = self.__weights.dims()[2] if len(self.__weights.dims()) > 2 else None
        dim3 = self.__weights.dims()[3] if len(self.__weights.dims()) > 3 else None
        # Make the lambda corresponding to self.__weights(0) == 0
        lambdat = af.constant(self.__lambda_param, dim0, dim1, dim2, dim3)

        # No regularization for bias weights
        lambdat[0, :] = 0

        # Get the prediction
        H = self.predict_proba(X)

        # Cost of misprediction
        Jerr = -1 * af.sum(Y * af.log(H) + (1 - Y) * af.log(1 - H), dim=0)

        # Regularization cost
        Jreg = 0.5 * af.sum(lambdat * self.__weights * self.__weights, dim=0)

        # Total cost
        J = (Jerr + Jreg) / m

        # Find the gradient of cost
        D = (H - Y)
        dJ = (af.matmulTN(X, D) + lambdat * self.__weights) / m

        return J, dJ


    def train(self, X, Y):
        # Initialize parameters to 0
        self.__weights = af.constant(0, X.dims()[1], Y.dims()[1])

        for i in range(self.__maxiter):
            # Get the cost and gradient
            J, dJ = self.cost(X, Y)
            err = af.max(af.abs(J))
            if err < self.__maxerr:
                print('Iteration {0:4d} Err: {1:4f}'.format(i + 1, err))
                print('Training converged')
                return self.__weights

            if self.__verbose and ((i+1) % 10 == 0):
                print('Iteration {0:4d} Err: {1:4f}'.format(i + 1, err))

            # Update the weights via gradient descent
            self.__weights = self.__weights - self.__alpha * dJ

        if self.__verbose:
            print('Training stopped after {0:d} iterations'.format(self.__maxiter))


    def eval(self):
        af.eval(self.__weights)
        af.sync()


class TestAFLogisticRegression(unittest.TestCase):

    def test(self):
        ############# Pure arrayfire-python example ###########

        # Determine number of classes if not provided
        num_classes = np.unique(iris.target).shape[0]

        # Convert numpy array to af array (and convert labels/targets from ints to
        # one-hot encodings)
        train_feats = af.from_ndarray(iris.data.astype('float32'))
        train_targets = af.from_ndarray(ints_to_onehots(iris.target.astype('uint32'), num_classes))
        test_feats = af.from_ndarray(iris.data.astype('float32'))
        test_targets = af.from_ndarray(ints_to_onehots(iris.target.astype('uint32'), num_classes))

        num_train = train_feats.dims()[0]
        num_test = test_feats.dims()[0]

        # Remove bias for now to match output with pure arrayfire example
        # Pure arrayfire example uses features with bias column for train and test
        # but we can't expect that for d3m's inputs in general
        # Add bias
        # train_bias = af.constant(1, num_train, 1)
        # test_bias = af.constant(1, num_test, 1)
        # train_feats = af.join(1, train_bias, train_feats)
        # test_feats = af.join(1, test_bias, test_feats)

        ref_clf = RefAfLogisticRegression(alpha=0.1,          # learning rate
                                          lambda_param = 1.0, # regularization constant
                                          maxerr=0.01,        # max error
                                          maxiter=1000,       # max iters
                                          verbose=False       # verbose mode
        )

        ref_clf.train(train_feats, train_targets)
        af_output = ref_clf.predict(test_feats)
        ref_output = af_output.to_ndarray()
        print('Completed reference calculation')

        ############# d3m-arrayfire example ###########

        hyperparams = AFLogisticRegression.Hyperparams.defaults()
        # Create the model object
        test_clf = AFLogisticRegression.AFLogisticRegression(hyperparams=hyperparams)
        train_set = iris.data
        targets = iris.target
        test_clf.set_training_data(inputs=train_set, outputs=targets)
        test_clf.fit()

        test_output = test_clf.produce(inputs=train_set)
        print('Completed test calculation')

        self.assertTrue(np.array_equal(ref_output, test_output.value))
        print('SUCCESS: Pure arrayfire-python output equals d3m-arrayfire output')

        # Testing get_params() and set_params()
        params = test_clf.get_params()
        test_clf.set_params(params=params)
        first_output = test_clf.produce(inputs=train_set)
        self.assertTrue(np.array_equal(ref_output, first_output.value))
        print('SUCCESS: get_params and set_params work')

        # pickle the params and hyperparams
        pickled_params = pickle.dumps(params)
        unpickled_params = pickle.loads(pickled_params)

        pickled_hyperparams = pickle.dumps(hyperparams)
        unpickled_hyperparams = pickle.loads(pickled_hyperparams)

        # Create a new object from pickled params and hyperparams
        pickle_params_clf = AFLogisticRegression.AFLogisticRegression(hyperparams=unpickled_hyperparams)
        pickle_params_clf.set_params(params=unpickled_params)
        pickle_params_clf.set_training_data(inputs=train_set, outputs=targets)
        pickle_params_clf.fit()
        pickle_params_output = pickle_params_clf.produce(inputs=train_set)

        # Check if outputs match
        self.assertTrue(np.array_equal(ref_output, pickle_params_output.value))
        # We want to test the running of the code without errors and not the correctness of it
        # since that is assumed to be tested by sklearn
        print("SUCCESS: Pickling and unpickling hyperparams and params work")

        # Pickling whole model doesn't work right now, probably because model
        # is stored in device memory and pickle cannot serialize it trivially
        # model = pickle.dumps(test_clf)
        # pickle_model_clf = pickle.loads(model)
        # pickle_model_output = pickle_model_clf.produce(inputs=train_set)
        # self.assertTrue(np.array_equal(ref_output, pickle_model_output.value))
        # print("SUCCESS: Pickling whole model works")


if __name__ == '__main__':
    unittest.main()
