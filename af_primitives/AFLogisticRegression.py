from typing import Any, Callable, List, Dict, Union, Optional, Sequence, Tuple
import typing
from numpy import ndarray
import numpy as np
from collections import OrderedDict
from scipy import sparse
import os

import arrayfire as af
from arrayfire.algorithm import max, imax, count, sum
from arrayfire.array import read_array, transpose
from arrayfire.arith import abs, sigmoid, log
from arrayfire.blas import matmul, matmulTN
from arrayfire.data import constant, join, moddims
from arrayfire.device import sync, eval
from arrayfire.interop import from_ndarray

from d3m.container.numpy import ndarray as d3m_ndarray
from d3m.container import DataFrame as d3m_dataframe
from d3m.metadata import hyperparams, params, base as metadata_base
from d3m import utils
from d3m.primitive_interfaces.base import CallResult, DockerContainer
import common_primitives.utils as common_utils
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.base import ProbabilisticCompositionalityMixin

Inputs = Union[d3m_dataframe, d3m_ndarray]
Outputs = d3m_ndarray

class Params(params.Params):
    n_classes_: Optional[int]
    n_features_: Optional[int]


class Hyperparams(hyperparams.Hyperparams):
    learning_rate = hyperparams.Hyperparameter[float](
        default=0.1,
        description='(alpha) Rate at which to update the weights at each iteration during gradient descent',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    reg_constant = hyperparams.Hyperparameter[float](
        default=1.0,
        description='(lambda) Weight decay',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    max_err = hyperparams.Hyperparameter[float](
        default=0.01,
        description='Maximum error',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    max_iter = hyperparams.Bounded[int](
        default=1000,
        lower=0,
        upper=None,
        description='Maximum number of iterations taken for the solver to converge.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    verbose = hyperparams.Hyperparameter[int](
        default=0,
        description="Controls the verbosity of the building process.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )


class AFLogisticRegression(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    Primitive implementing LogisticRegression using the ArrayFire library
    """

    __author__ = "ArrayFire"
    metadata = metadata_base.PrimitiveMetadata({
         "algorithm_types": [metadata_base.PrimitiveAlgorithmType.LOGISTIC_REGRESSION, ],
         "name": "af.LogisticRegression",
         "primitive_family": metadata_base.PrimitiveFamily.REGRESSION,
         "python_path": "d3m.primitives.af_primitives.AFLogisticRegression",
         #"source": {'name': 'ArrayFire', 'contact': 'mailto:support@arrayfire.com', 'uris': ['https://gitlab.com/arrayfire/arrayfire']},
         "version": "0.0.1",
         "id": "73dff093-f8fe-4e9e-a5af-88a7e4398a43"
         #TODO:
         # 'installation': [
         #                {'type': metadata_base.PrimitiveInstallationType.PIP,
         #                   'package_uri': 'git+https://gitlab.com/datadrivendiscovery/sklearn-wrap.git@{git_commit}#egg=sklearn_wrap'.format(
         #                       git_commit=utils.current_git_commit(os.path.dirname(__file__)),
         #                    ),
         #                }]
    })

    def __init__(self, *,
                 hyperparams: Hyperparams,
                 random_seed: int = 0,) -> None:

        super().__init__(hyperparams=hyperparams, random_seed=random_seed)

        self._alpha = self.hyperparams['learning_rate']
        self._lambda_param = self.hyperparams['reg_constant']
        self._maxerr = self.hyperparams['max_err']
        self._maxiter = self.hyperparams['max_iter']
        self._verbose = bool(self.hyperparams['verbose'])

        self._training_inputs = None
        self._training_outputs = None
        self._fitted = False
        self._weights = None

        self._classes = None
        self._n_classes = 0
        self._n_features = 0

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        self._training_inputs = inputs
        self._training_outputs = outputs
        self._fitted = False

    def _accuracy(self, predicted, target):
        _, tlabels = af.imax(target, 1)
        _, plabels = af.imax(predicted, 1)
        return 100 * af.count(plabels == tlabels) / tlabels.elements()

    def _abserr(self, predicted, target):
        return 100 * af.sum(af.abs(predicted - target)) / predicted.elements()

    def _predict_proba(self, X, Weights):
        # print('X: {}'.format(X.dtype()))
        # print('Weights: {}'.format(Weights.dtype()))
        Z = af.matmul(X, Weights)
        return af.sigmoid(Z)

    def _predict_log_proba(self, X, Weights):
        return af.log(self._predict_proba(X, Weights))

    def _predict(self, X, Weights):
        probs = self._predict_proba(X, Weights)
        _, classes = af.imax(probs, 1)
        return classes

    def _cost(self, Weights, X, Y, lambda_param=1.0):
        # Number of samples
        m = Y.dims()[0]

        dim0 = Weights.dims()[0]
        dim1 = Weights.dims()[1] if len(Weights.dims()) > 1 else None
        dim2 = Weights.dims()[2] if len(Weights.dims()) > 2 else None
        dim3 = Weights.dims()[3] if len(Weights.dims()) > 3 else None
        # Make the lambda corresponding to Weights(0) == 0
        lambdat = af.constant(lambda_param, dim0, dim1, dim2, dim3)

        # No regularization for bias weights
        lambdat[0, :] = 0

        # Get the prediction
        H = self._predict_proba(X, Weights)

        # Cost of misprediction
        Jerr = -1 * af.sum(Y * af.log(H) + (1 - Y) * af.log(1 - H), dim=0)

        # Regularization cost
        # penalty_norm = None
        # if self.hyperparams['penalty'] == 'l2':
        #     penalty_norm = Weights * Weights
        # else:
        #     penalty_norm = af.abs(Weights)
        penalty_norm = Weights * Weights
        Jreg = 0.5 * af.sum(lambdat * penalty_norm, dim=0)

        # Total cost
        J = (Jerr + Jreg) / m

        # Find the gradient of cost
        D = (H - Y)
        dJ = (af.matmulTN(X, D) + lambdat * Weights) / m

        return J, dJ

    def _ints_to_onehots(self, digits, num_classes):
        # print('num_classes: {}'.format(num_classes))
        onehots = np.zeros((digits.shape[0], num_classes), dtype='float32')
        onehots[np.arange(digits.shape[0]), digits] = 1
        return onehots

    def _train(self, X, Y, alpha=0.1, lambda_param=1.0, maxerr=0.01, maxiter=1000):
        # Initialize parameters to 0
        Weights = af.constant(0, X.dims()[1], Y.dims()[1])

        for i in range(maxiter):
            # Get the cost and gradient
            J, dJ = self._cost(Weights, X, Y, lambda_param)
            err = af.max(af.abs(J))
            if err < maxerr:
                return Weights

            # Update the weights via gradient descent
            Weights = Weights - alpha * dJ

        return Weights

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        # Assume that inputs at this point are already ndarrays
        if self._fitted:
            return CallResult(None)

        if self._training_inputs is None or self._training_outputs is None:
            raise ValueError("Missing training data.")

        # Assume training input data is an ndarray
        training_inputs = self._training_inputs
        training_outputs = self._training_outputs

        if self._n_classes == 0:
            self._n_classes = np.unique(training_outputs).shape[0]
        self._n_features = training_inputs.shape[1]

        # Flatten output if needed
        shape = training_outputs.shape
        if len(shape) == 2 and shape[1] == 1:
            training_outputs = numpy.ravel(training_outputs)

        # Convert ndarray to af array
        train_images = af.from_ndarray(training_inputs)
        # print('training_outputs: {}'.format(training_outputs.shape))
        # print(training_outputs[:5])
        # print('training_outputs unique: {}'.format(np.unique(training_outputs)))
        train_targets = af.from_ndarray(
            self._ints_to_onehots(training_outputs, self._n_classes)
        )

        # Flatten training samples if they're multidimensional
        num_train = train_images.dims()[0]
        feature_length = int(train_images.elements() / num_train);
        train_feats = af.moddims(train_images, num_train, feature_length)

        # Add bias
        train_bias = af.constant(1, num_train, 1)
        train_feats = af.join(1, train_bias, train_feats)

        # Start training
        self._weights = self._train(train_feats, train_targets,
                                    self._alpha,
                                    self._lambda_param,
                                    self._maxerr,
                                    self._maxiter
        )

        self._fitted = True

        return CallResult(None)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        # We may have to adjust the dimensions when doing this conversion
        af_inputs = af.from_ndarray(inputs)
        # print('af_inputs: {}'.format(af_inputs.dims()))
        # print('weights: {}'.format(self._weights.dims()))
        af_output = self._predict(af_inputs, self._weights[:4, :])
        output = af_output.to_ndarray()
        return CallResult(d3m_ndarray(output))

    def get_params(self) -> Params:
        if not self._fitted:
            raise ValueError("Fit not performed.")

        return Params(
            n_classes_ = self._n_classes,
            n_features_ = self._n_features
        )


    def set_params(self, *, params: Params) -> None:
        self._n_classes = params['n_classes_']
        self._n_features = params['n_features_']


