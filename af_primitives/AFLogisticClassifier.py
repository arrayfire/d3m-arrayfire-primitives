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

Inputs = Union[d3m_DataFrame, d3m_ndarray]
Outputs = d3m_ndarray

class Params(params.Params):
    coef_: Optional[ndarray]
    intercept_: Optional[ndarray]
    n_iter_: Optional[ndarray]
    classes_: Optional[ndarray]
    target_names_: Optional[Sequence[Any]]
    training_indices_: Optional[Sequence[int]]

class Hyperparams(hyperparams.Hyperparams):
    learning_rate = hyperparams.Hyperparameter[float](
        default=0.1,
        description='Rate at which to update the weights at each iteration during gradient descent',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    max_error = hyperparams.Hyperparameter[float](
        default=0.01,
        description='Maximum error of weights before training is declared to have converged',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    penalty = hyperparams.Enumeration[str](
        values=['l1', 'l2'],
        default='l2',
        description='Used to specify the norm used in the penalization.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    fit_intercept = hyperparams.UniformBool(
        default=True,
        description='Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    class_weight = hyperparams.Union(
        configuration=OrderedDict({
            'str': hyperparams.Enumeration[str](
                default='balanced',
                values=['balanced'],
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            ),
            'none': hyperparams.Hyperparameter[None](
                default=None,
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            )
        }),
        default='none',
        description='Weights associated with classes in the form ``{class_label: weight}``. If not given, all classes are supposed to have weight one.  The "balanced" mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as ``n_samples / (n_classes * np.bincount(y))``.  Note that these weights will be multiplied with sample_weight (passed through the fit method) if sample_weight is specified.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    max_iter = hyperparams.Bounded[int](
        default=100,
        lower=0,
        upper=None,
        description='Maximum number of iterations taken for the solver to converge.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    C = hyperparams.Hyperparameter[float](
        default=1.0,
        description='Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    multi_class = hyperparams.Enumeration[str](
        values=['multinomial'],
        default='multinomial',
        description='Multiclass option be either \'multinomial\'. The loss minimised is the multinomial loss fit across the entire probability distribution.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    use_input_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to use as training input. If any specified column cannot be parsed, it is skipped.",
    )
    use_output_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to use as training target. If any specified column cannot be parsed, it is skipped.",
    )
    exclude_input_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to not use as training inputs. Applicable only if \"use_columns\" is not provided.",
    )
    exclude_output_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to not use as training target. Applicable only if \"use_columns\" is not provided.",
    )
    return_result = hyperparams.Enumeration(
        values=['append', 'replace', 'new'],
        default='new',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Should parsed columns be appended, should they replace original columns, or should only parsed columns be returned? This hyperparam is ignored if use_semantic_types is set to false.",
    )
    use_semantic_types = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Controls whether semantic_types metadata will be used for filtering columns in input dataframe. Setting this to false makes the code ignore return_result and will produce only the output dataframe"
    )
    add_index_columns = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Also include primary index columns if input data has them. Applicable only if \"return_result\" is set to \"new\".",
    )


class af_LogisticRegression(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams],
                           ProbabilisticCompositionalityMixin[Inputs, Outputs, Params, Hyperparams]):
    """
    Primitive implementing LogisticRegression using ArrayFire gpu library
    """

    __author__ = "ArrayFire"
    metadata = metadata_base.PrimitiveMetadata({ 
         "algorithm_types": [metadata_base.PrimitiveAlgorithmType.LOGISTIC_REGRESSION, ],
         "name": "af.LogisticClassifier",
         "primitive_family": metadata_base.PrimitiveFamily.CLASSIFICATION,
         "python_path": "d3m.primitives.af_primitives.AFLogisticClassifier",
         #"source": {'name': 'ArrayFire', 'contact': 'mailto:support@arrayfire.com', 'uris': ['https://gitlab.com/arrayfire/arrayfire']},
         "version": "0.0.1",
         "id": "73dff093-f8fe-4e9e-a5af-88a7e4398a43"
         #TODO:
         #'installation': [
                        #{'type': metadata_base.PrimitiveInstallationType.PIP,
                           #'package_uri': 'git+https://gitlab.com/datadrivendiscovery/sklearn-wrap.git@{git_commit}#egg=sklearn_wrap'.format(
                               #git_commit=utils.current_git_commit(os.path.dirname(__file__)),
                            #),
                        #}]
    })

    def __init__(self, *,
                 hyperparams: Hyperparams,
                 random_seed: int = 0,
                 docker_containers: Dict[str, DockerContainer] = None,
                 _verbose: int = 0) -> None:

        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

        # # False
        # self._clf = LogisticRegression(
        #       penalty=self.hyperparams['penalty'],
        #       dual=self.hyperparams['dual'],
        #       fit_intercept=self.hyperparams['fit_intercept'],
        #       intercept_scaling=self.hyperparams['intercept_scaling'],
        #       class_weight=self.hyperparams['class_weight'],
        #       max_iter=self.hyperparams['max_iter'],
        #       solver=self.hyperparams['solver'],
        #       tol=self.hyperparams['tol'],
        #       C=self.hyperparams['C'],
        #       multi_class=self.hyperparams['multi_class'],
        #       warm_start=self.hyperparams['warm_start'],
        #       n_jobs=self.hyperparams['n_jobs'],
        #       random_state=self.random_seed,
        #       verbose=_verbose
        # )

        self._training_inputs = None
        self._training_outputs = None
        self._target_names = None
        self._training_indices = None
        self._fitted = False
        self._Weights = None

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        self._training_inputs, self._training_indices = self._get_columns_to_fit(inputs, self.hyperparams)
        self._training_outputs, self._target_names = self._get_targets(outputs, self.hyperparams)
        self._fitted = False

    def _accuracy(self, predicted, target):
        _, tlabels = af.imax(target, 1)
        _, plabels = af.imax(predicted, 1)
        return 100 * af.count(plabels == tlabels) / tlabels.elements()

    def _abserr(self, predicted, target):
        return 100 * af.sum(af.abs(predicted - target)) / predicted.elements()

    def _predict_proba(self, X, Weights):
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
        penalty_norm = None
        if self.hyperparams['penalty'] == 'l2':
            penalty_norm = Weights * Weights
        else:
            penalty_norm = af.abs(Weights)
        Jreg = 0.5 * af.sum(lambdat * penalty_norm, dim=0)

        # Total cost
        J = (Jerr + Jreg) / m

        # Find the gradient of cost
        D = (H - Y)
        dJ = (af.matmulTN(X, D) + lambdat * Weights) / m

        return J, dJ

    def _ints_to_onehots(self, digits, num_classes):
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
            Weights = Weights - alpha * d

        return Weights

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        # Assume that inputs at this point are already ndarrays
        if self._fitted:
            return CallResult(None)

        if self._training_inputs is None or self._training_outputs is None:
            raise ValueError("Missing training data.")

        # self._clf.fit(self._training_inputs, sk_training_output)

        # Assume training input data is an ndarray
        training_inputs = self._training_inputs
        training_outputs = self._training_outputs.values
        params = self.get_params()

        # Flatten output if needed
        shape = training_outputs.shape
        if len(shape) == 2 and shape[1] == 1:
            training_outputs = numpy.ravel(training_outputs)

        # Convert ndarray to af array
        train_images = af.from_ndarray(training_inputs)
        train_targets = af.from_ndarray(
            self._ints_to_onehots(training_outputs,
                                  len(params['classes_'])
            )
        )

        # Flatten training samples if they're multidimensional
        num_train = train_images.dims()[0]
        feature_length = int(train_images.elements() / num_train);
        train_feats = af.moddims(train_images, num_train, feature_length)

        # Add bias
        train_bias = af.constant(1, num_train, 1)
        train_feats = af.join(1, train_bias, train_feats)

        # Start training
        self._Weights = self._train(train_feats, train_targets,
                                    self.hyperparams['learning_rate'],
                                    self.hyperparams['C'],
                                    self.hyperparams['max_error'],
                                    self.hyperparams['max_iter']
        )

        self._fitted = True

        return CallResult(None)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        # We may have to adjust the dimensions when doing this conversion
        af_inputs = af.from_ndarray(inputs.values)
        # if self.hyperparams['use_semantic_types']:
        #     af_inputs = inputs.iloc[:, self._training_indices]
        af_output = self._predict(af_inputs, self._Weights)
        af_output = af_output.to_ndarray()
        output = d3m_dataframe(af_output, generate_metadata=False, source=self)
        output.metadata = inputs.metadata.clear(source=self, for_value=output, generate_metadata=True)
        output.metadata = self._add_target_semantic_types(metadata=output.metadata, target_names=self._target_names, source=self)
        outputs = common_utils.combine_columns(return_result=self.hyperparams['return_result'],
                                               add_index_columns=self.hyperparams['add_index_columns'],
                                               inputs=inputs, column_indices=self._training_indices, columns_list=[output], source=self)

        return CallResult(outputs)

    def get_params(self) -> Params:
        if not self._fitted:
            return Params(
                coef_=None,
                intercept_=None,
                n_iter_=None,
                classes_=None,
                training_indices_=self._training_indices,
                target_names_=self._target_names
            )

        return Params(
            coef_=getattr(self._clf, 'coef_', None),
            intercept_=getattr(self._clf, 'intercept_', None),
            n_iter_=getattr(self._clf, 'n_iter_', None),
            classes_=getattr(self._clf, 'classes_', None),
            training_indices_=self._training_indices,
            target_names_=self._target_names
        )

    def set_params(self, *, params: Params) -> None:
        self._clf.coef_ = params['coef_']
        self._clf.intercept_ = params['intercept_']
        self._clf.n_iter_ = params['n_iter_']
        self._clf.classes_ = params['classes_']
        self._training_indices = params['training_indices_']
        self._target_names = params['target_names_']
        self._fitted = False

        if params['coef_'] is not None:
            self._fitted = True
        if params['intercept_'] is not None:
            self._fitted = True
        if params['n_iter_'] is not None:
            self._fitted = True
        if params['classes_'] is not None:
            self._fitted = True

    def log_likelihoods(self, *,
                    outputs: Outputs,
                    inputs: Inputs,
                    timeout: float = None,
                    iterations: int = None) -> CallResult[Sequence[float]]:
        inputs = inputs.values  # Get ndarray
        outputs = outputs.values

        af_inputs = af.from_ndarray(inputs)
        af_outputs = af.from_ndarray(outputs)
        log_proba = self._predict_log_proba(inputs, self._Weights)
        return CallResult(log_proba.to_ndarray()[:, outputs])

    @classmethod
    def _get_columns_to_fit(cls, inputs: Inputs, hyperparams: Hyperparams):
        if not hyperparams['use_semantic_types']:
            return inputs, list(range(len(inputs.columns)))

        inputs_metadata = inputs.metadata

        def can_produce_column(column_index: int) -> bool:
            return cls._can_produce_column(inputs_metadata, column_index, hyperparams)

        columns_to_produce, columns_not_to_produce = common_utils.get_columns_to_use(inputs_metadata,
                                                                             use_columns=hyperparams['use_input_columns'],
                                                                             exclude_columns=hyperparams['exclude_input_columns'],
                                                                             can_use_column=can_produce_column)
        return inputs.iloc[:, columns_to_produce], columns_to_produce
        # return columns_to_produce

    @classmethod
    def _can_produce_column(cls, inputs_metadata: metadata_base.DataMetadata, column_index: int, hyperparams: Hyperparams) -> bool:
        column_metadata = inputs_metadata.query((metadata_base.ALL_ELEMENTS, column_index))

        accepted_structural_types = (int, float, numpy.integer, numpy.float64)
        accepted_semantic_types = set()
        accepted_semantic_types.add("https://metadata.datadrivendiscovery.org/types/Attribute")
        if not issubclass(column_metadata['structural_type'], accepted_structural_types):
            return False

        semantic_types = set(column_metadata.get('semantic_types', []))

        if len(semantic_types) == 0:
            cls.logger.warning("No semantic types found in column metadata")
            return False
        # Making sure all accepted_semantic_types are available in semantic_types
        if len(accepted_semantic_types - semantic_types) == 0:
            return True

        return False

    @classmethod
    def _get_targets(cls, data: d3m_dataframe, hyperparams: Hyperparams):
        if not hyperparams['use_semantic_types']:
            return data, []
        target_names = []
        target_column_indices = []
        metadata = data.metadata

        def can_produce_column(column_index: int) -> bool:
            accepted_semantic_types = set()
            accepted_semantic_types.add("https://metadata.datadrivendiscovery.org/types/TrueTarget")
            column_metadata = metadata.query((metadata_base.ALL_ELEMENTS, column_index))
            semantic_types = set(column_metadata.get('semantic_types', []))
            if len(semantic_types) == 0:
                cls.logger.warning("No semantic types found in column metadata")
                return False
            # Making sure all accepted_semantic_types are available in semantic_types
            if len(accepted_semantic_types - semantic_types) == 0:
                return True
            return False

        target_column_indices, target_columns_not_to_produce = common_utils.get_columns_to_use(metadata,
                                                                             use_columns=hyperparams['use_output_columns'],
                                                                             exclude_columns=hyperparams['exclude_output_columns'],
                                                                             can_use_column=can_produce_column)

        for column_index in target_column_indices:
            if column_index is metadata_base.ALL_ELEMENTS:
                continue
            column_index = typing.cast(metadata_base.SimpleSelectorSegment, column_index)
            column_metadata = metadata.query((metadata_base.ALL_ELEMENTS, column_index))
            target_names.append(column_metadata.get('name', str(column_index)))

        targets = data.iloc[:, target_column_indices]
        return targets, target_names

    @classmethod
    def _add_target_semantic_types(cls, metadata: metadata_base.DataMetadata,
                            source: typing.Any,  target_names: List = None,) -> metadata_base.DataMetadata:
        for column_index in range(metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']):
            metadata = metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, column_index),
                                                  'https://metadata.datadrivendiscovery.org/types/Target',
                                                  source=source)
            metadata = metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, column_index),
                                                  'https://metadata.datadrivendiscovery.org/types/PredictedTarget',
                                                  source=source)
            if target_names:
                metadata = metadata.update((metadata_base.ALL_ELEMENTS, column_index), {
                    'name': target_names[column_index],
                }, source=source)
        return metadata

SKLogisticRegression.__doc__ = LogisticRegression.__doc__
