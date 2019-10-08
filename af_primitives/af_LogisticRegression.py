import os
from typing import Any, Callable, List, Dict, Union, Optional, Sequence, Tuple
from collections import OrderedDict
import numpy as np
from numpy import ndarray

from d3m import exceptions
from d3m import utils
from d3m.base import utils as base_utils
from d3m.container import DataFrame as d3m_dataframe
from d3m.exceptions import PrimitiveNotFittedError
from d3m.metadata import hyperparams, params, base as metadata_base
from d3m.primitive_interfaces.base import CallResult, DockerContainer
from d3m.primitive_interfaces.base import ProbabilisticCompositionalityMixin
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase

import arrayfire as af
from arrayfire.algorithm import max, imax, count, sum
from arrayfire.arith import abs, sigmoid, log
from arrayfire.array import read_array, transpose
from arrayfire.blas import matmul, matmulTN
from arrayfire.data import constant, join, moddims
from arrayfire.device import sync, eval
from arrayfire.interop import from_ndarray


Inputs = d3m_dataframe
Outputs = d3m_dataframe


class Params(params.Params):
    classes_: Optional[ndarray]
    input_column_names: Optional[Any]
    target_names_: Optional[Sequence[Any]]
    training_indices_: Optional[Sequence[int]]
    target_column_indices_: Optional[Sequence[int]]
    target_columns_metadata_: Optional[List[OrderedDict]]


class Hyperparams(hyperparams.Hyperparams):
    penalty = hyperparams.Enumeration[str](
        values=['l1', 'l2'],
        default='l2',
        description='Used to specify the norm used in the penalization. The \'newton-cg\', \'sag\' and \'lbfgs\' solvers support only l2 penalties.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    use_inputs_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to use as training input. If any specified column cannot be parsed, it is skipped.",
    )
    use_outputs_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to use as training target. If any specified column cannot be parsed, it is skipped.",
    )
    exclude_inputs_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to not use as training inputs. Applicable only if \"use_columns\" is not provided.",
    )
    exclude_outputs_columns = hyperparams.Set(
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
    error_on_no_input = hyperparams.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Throw an exception if no input column is selected/provided. Defaults to true to behave like sklearn. To prevent pipelines from breaking set this to False.",
    )

    return_semantic_type = hyperparams.Enumeration[str](
        values=['https://metadata.datadrivendiscovery.org/types/Attribute', 'https://metadata.datadrivendiscovery.org/types/ConstructedAttribute', 'https://metadata.datadrivendiscovery.org/types/PredictedTarget'],
        default='https://metadata.datadrivendiscovery.org/types/PredictedTarget',
        description='Decides what semantic type to attach to generated output',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )
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
        description='Controls the verbosity of the building process.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )


class af_LogisticRegression(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams],
                            ProbabilisticCompositionalityMixin[Inputs, Outputs, Params, Hyperparams]):
    '''
    Primitive implementing LogisticRegression using the ArrayFire library
    '''

    __author__ = 'ArrayFire'
    metadata = metadata_base.PrimitiveMetadata({
        'name': 'ArrayFire Logistic Regression',
        'source': {
            'name': 'ArrayFire',
            'contact': 'mailto:support@arrayfire.com',
            'uris': ['https://gitlab.com/syurkevi/d3m-arrayfire-primitives']},
        'id': '25b08bb7-12f0-4447-a75b-5856ead6227e',
        'version': '0.1.0',
        'python_path': 'd3m.primitives.classification.logistic_regression.ArrayFire',
        'keywords' : ['arrayfire', 'logistic regression', 'logistic regressor'],
        'installation': [
            {'type': metadata_base.PrimitiveInstallationType.PIP,
             'package_uri': 'git+https://gitlab.com/syurkevi/d3m-arrayfire-primitives@{git_commit}#egg=af_primitives'.format(
                 git_commit=utils.current_git_commit(os.path.dirname(__file__)),
             ),
            }],
        'algorithm_types': [ metadata_base.PrimitiveAlgorithmType.LOGISTIC_REGRESSION, ],
        'primitive_family': metadata_base.PrimitiveFamily.CLASSIFICATION,
        'hyperparameters_to_tune': ['learning_rate', 'reg_constant', 'max_err', 'max_iter']
    })


    def __init__(self, *,
                 hyperparams: Hyperparams,
                 random_seed: int = 0,
                 docker_containers: Dict[str, DockerContainer] = None,
                 _verbose: int = 0) -> None:

        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

        self._learning_rate = self.hyperparams['learning_rate']
        self._reg_constant = self.hyperparams['reg_constant']
        self._penalty = self.hyperparams['penalty']
        self._max_err = self.hyperparams['max_err']
        self._max_iter = self.hyperparams['max_iter']
        self._verbose = bool(self.hyperparams['verbose'])
        self._classes = None
        self._n_classes = 0
        self._label_offset = 0
        self._max_feature_value = 0
        self._max_feature_value_defined = False

        self._inputs = None
        self._outputs = None
        self._training_inputs = None
        self._training_outputs = None
        self._target_names = None
        self._training_indices = None
        self._target_column_indices = None
        self._target_columns_metadata: List[OrderedDict] = None
        self._input_column_names = None
        self._weights = None
        self._fitted = False
        self._new_training_data = False


    @classmethod
    def _accuracy(self, predicted, target):
        _, tlabels = af.imax(target, 1)
        _, plabels = af.imax(predicted, 1)
        return 100 * af.count(plabels == tlabels) / tlabels.elements()


    @classmethod
    def _abserr(self, predicted, target):
        return 100 * af.sum(af.abs(predicted - target)) / predicted.elements()


    @classmethod
    def _predict_proba(self, X, Weights):
        Z = af.matmul(X, Weights)
        return af.sigmoid(Z)


    @classmethod
    def _predict_log_proba(self, X, Weights):
        return af.log(self._predict_proba(X, Weights))


    @classmethod
    def _predict(self, X, Weights):
        probs = self._predict_proba(X, Weights)
        _, classes = af.imax(probs, 1)
        classes = classes + self._label_offset
        return classes


    @classmethod
    def _cost(self, Weights, X, Y, reg_constant, penalty):
        # Number of samples
        m = Y.dims()[0]

        dim0 = Weights.dims()[0]
        dim1 = Weights.dims()[1] if len(Weights.dims()) > 1 else None
        dim2 = Weights.dims()[2] if len(Weights.dims()) > 2 else None
        dim3 = Weights.dims()[3] if len(Weights.dims()) > 3 else None

        # Make the lambda corresponding to Weights(0) == 0
        lambdat = af.constant(reg_constant, dim0, dim1, dim2, dim3)

        # No regularization for bias weights
        lambdat[0, :] = 0

        # Get the prediction
        H = self._predict_proba(X, Weights)

        # Cost of misprediction
        Jerr = -1 * af.sum(Y * af.log(H) + (1 - Y) * af.log(1 - H), dim=0)

        # Regularization cost
        penalty_norm = None
        if penalty == 'l2':
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


    @classmethod
    def _ints_to_onehots(self, digits, num_classes):
        # Need labels to start with 0, but some datasets might start with 1 or other numbers
        self._label_offset = np.amin(digits)
        onehots = np.zeros((digits.shape[0], num_classes), dtype='float32')
        onehots[np.arange(digits.shape[0]), digits - self._label_offset] = 1
        return onehots


    @classmethod
    def _train(self, X, Y, alpha=0.1, lambda_param=1.0, penalty='l2', maxerr=0.01, maxiter=1000):
        # Initialize parameters to 0
        Weights = af.constant(0, X.dims()[1], Y.dims()[1])

        for i in range(maxiter):
            # Get the cost and gradient
            J, dJ = self._cost(Weights, X, Y, lambda_param, penalty)
            err = af.max(af.abs(J))
            if err < maxerr:
                Weights = Weights[:-1] # Remove bias weights
                return Weights

            # Update the weights via gradient descent
            Weights = Weights - alpha * dJ

        # Remove bias weights
        Weights = Weights[:-1]

        return Weights


    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        self._inputs = inputs
        self._outputs = outputs
        self._fitted = False
        self._new_training_data = True


    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        if self._inputs is None or self._outputs is None:
            raise ValueError("Missing training data.")

        if not self._new_training_data:
            return CallResult(None)
        self._new_training_data = False

        self._training_inputs, self._training_indices = self._get_columns_to_fit(self._inputs, self.hyperparams)
        self._training_outputs, self._target_names, self._target_column_indices = self._get_targets(self._outputs, self.hyperparams)
        self._input_column_names = self._training_inputs.columns

        if len(self._training_indices) > 0 and len(self._target_column_indices) > 0:
            self._target_columns_metadata = self._get_target_columns_metadata(self._training_outputs.metadata, self.hyperparams)
            sk_training_output = self._training_outputs.values

            shape = sk_training_output.shape
            if len(shape) == 2 and shape[1] == 1:
                sk_training_output = np.ravel(sk_training_output)

            # Assume training input data is an ndarray
            training_inputs = self._training_inputs.values.astype('float32')
            training_outputs = sk_training_output.astype('uint32')

            if self._n_classes == 0:
                # Assume that class labels are integers and nonnegative
                self._n_classes = np.amax(training_outputs).astype('uint32').item() + 1


            # Convert ndarray to af array
            train_feats = af.from_ndarray(training_inputs)
            train_targets = af.from_ndarray(
                self._ints_to_onehots(training_outputs, self._n_classes)
            )
            num_train = train_feats.dims()[0]

            # Normalize feature values
            self._max_feature_value = af.max(train_feats)
            self._max_feature_value_defined = True
            train_feats = train_feats / self._max_feature_value

            # Add bias feature
            train_bias = af.constant(1, num_train, 1)
            train_feats = af.join(1, train_bias, train_feats)

            # Start training
            self._weights = self._train(train_feats, train_targets,
                                        self._learning_rate,
                                        self._reg_constant,
                                        self._penalty,
                                        self._max_err,
                                        self._max_iter
            )

            self._fitted = True
        else:
            if self.hyperparams['error_on_no_input']:
                raise RuntimeError("No input columns were selected")
            self.logger.warn("No input columns were selected")

        return CallResult(None)


    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        if not self._fitted:
            raise PrimitiveNotFittedError("Primitive not fitted.")

        sk_inputs, columns_to_use = self._get_columns_to_fit(inputs, self.hyperparams)

        output = []
        if len(sk_inputs.columns):
            af_inputs = af.from_ndarray(sk_inputs.values.astype('float32'))

            # Normalize feature values
            if not self._max_feature_value_defined:
                self._max_feature_value = af.max(train_feats)
            af_inputs = af_inputs / self._max_feature_value

            af_output = self._predict(af_inputs, self._weights)
            ndarray_output = af_output.to_ndarray()

            output = self._wrap_predictions(inputs, ndarray_output)
            output.columns = self._target_names
            output = [output]
        else:
            if self.hyperparams['error_on_no_input']:
                raise RuntimeError("No input columns were selected")
            self.logger.warn("No input columns were selected")

        outputs = base_utils.combine_columns(return_result=self.hyperparams['return_result'],
                                             add_index_columns=self.hyperparams['add_index_columns'],
                                             inputs=inputs, column_indices=self._target_column_indices,
                                             columns_list=output)

        return CallResult(outputs)


    def get_params(self) -> Params:
        if not self._fitted:
            return Params(
                classes_=None,
                input_column_names=self._input_column_names,
                training_indices_=self._training_indices,
                target_names_=self._target_names,
                target_column_indices_=self._target_column_indices,
                target_columns_metadata_=self._target_columns_metadata
            )

        return Params(
            classes_=self._classes,
            input_column_names=self._input_column_names,
            training_indices_=self._training_indices,
            target_names_=self._target_names,
            target_column_indices_=self._target_column_indices,
            target_columns_metadata_=self._target_columns_metadata
        )


    def set_params(self, *, params: Params) -> None:
        self._classes_ = params['classes_']
        self._input_column_names = params['input_column_names']
        self._training_indices = params['training_indices_']
        self._target_names = params['target_names_']
        self._target_column_indices = params['target_column_indices_']
        self._target_columns_metadata = params['target_columns_metadata_']


    def log_likelihoods(self, *,
                        outputs: Outputs,
                        inputs: Inputs,
                        timeout: float = None,
                        iterations: int = None) -> CallResult[Sequence[float]]:
        inputs = inputs.iloc[:, self._training_indices]  # Get ndarray
        outputs = outputs.iloc[:, self._target_column_indices]

        if len(inputs.columns) and len(outputs.columns):

            if outputs.shape[1] != self._n_classes:
                raise exceptions.InvalidArgumentValueError("\"outputs\" argument does not have the correct number of target columns.")

            log_proba = self._predict_log_proba(inputs, self._weights)

            # Making it always a list, even when only one target.
            if self._n_classes == 1:
                log_proba = [log_proba]
                classes = [self._classes_]
            else:
                classes = self._classes_

            samples_length = inputs.shape[0]

            log_likelihoods = []
            for k in range(self._n_classes):
                # We have to map each class to its internal (numerical) index used in the learner.
                # This allows "outputs" to contain string classes.
                outputs_column = outputs.iloc[:, k]
                classes_map = pandas.Series(np.arange(len(classes[k])), index=classes[k])
                mapped_outputs_column = outputs_column.map(classes_map)

                # For each target column (column in "outputs"), for each sample (row) we pick the log
                # likelihood for a given class.
                log_likelihoods.append(log_proba[k][np.arange(samples_length), mapped_outputs_column])

            results = d3m_dataframe(dict(enumerate(log_likelihoods)), generate_metadata=True)
            results.columns = outputs.columns

            for k in range(self._n_classes):
                column_metadata = outputs.metadata.query_column(k)
                if 'name' in column_metadata:
                    results.metadata = results.metadata.update_column(k, {'name': column_metadata['name']})

        else:
            results = d3m_dataframe(generate_metadata=True)

        return CallResult(results)


    @classmethod
    def _get_columns_to_fit(cls, inputs: Inputs, hyperparams: Hyperparams):
        if not hyperparams['use_semantic_types']:
            return inputs, list(range(len(inputs.columns)))

        inputs_metadata = inputs.metadata

        def can_produce_column(column_index: int) -> bool:
            return cls._can_produce_column(inputs_metadata, column_index, hyperparams)

        columns_to_produce, columns_not_to_produce = base_utils.get_columns_to_use(inputs_metadata,
                                                                                   use_columns=hyperparams['use_inputs_columns'],
                                                                                   exclude_columns=hyperparams['exclude_inputs_columns'],
                                                                                   can_use_column=can_produce_column)
        return inputs.iloc[:, columns_to_produce], columns_to_produce
        # return columns_to_produce


    @classmethod
    def _can_produce_column(cls, inputs_metadata: metadata_base.DataMetadata, column_index: int, hyperparams: Hyperparams) -> bool:
        column_metadata = inputs_metadata.query((metadata_base.ALL_ELEMENTS, column_index))

        accepted_structural_types = (int, float, np.integer, np.float64)
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
            return data, list(data.columns), list(range(len(data.columns)))

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

        target_column_indices, target_columns_not_to_produce = base_utils.get_columns_to_use(metadata,
                                                                                             use_columns=hyperparams[
                                                                                                 'use_outputs_columns'],
                                                                                             exclude_columns=
                                                                                             hyperparams[
                                                                                                 'exclude_outputs_columns'],
                                                                                             can_use_column=can_produce_column)

        targets = []
        if target_column_indices:
            targets = data.select_columns(target_column_indices)
        target_column_names = []
        for idx in target_column_indices:
            target_column_names.append(data.columns[idx])
        return targets, target_column_names, target_column_indices


    @classmethod
    def _get_target_columns_metadata(cls, outputs_metadata: metadata_base.DataMetadata, hyperparams) -> List[OrderedDict]:
        outputs_length = outputs_metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']

        target_columns_metadata: List[OrderedDict] = []
        for column_index in range(outputs_length):
            column_metadata = OrderedDict(outputs_metadata.query_column(column_index))

            # Update semantic types and prepare it for predicted targets.
            semantic_types = set(column_metadata.get('semantic_types', []))
            semantic_types_to_remove = set(["https://metadata.datadrivendiscovery.org/types/TrueTarget","https://metadata.datadrivendiscovery.org/types/SuggestedTarget",])
            add_semantic_types = set(["https://metadata.datadrivendiscovery.org/types/PredictedTarget",])
            add_semantic_types.add(hyperparams["return_semantic_type"])
            semantic_types = semantic_types - semantic_types_to_remove
            semantic_types = semantic_types.union(add_semantic_types)
            column_metadata['semantic_types'] = list(semantic_types)

            target_columns_metadata.append(column_metadata)

        return target_columns_metadata


    @classmethod
    def _update_predictions_metadata(cls, inputs_metadata: metadata_base.DataMetadata, outputs: Optional[Outputs],
                                     target_columns_metadata: List[OrderedDict]) -> metadata_base.DataMetadata:
        outputs_metadata = metadata_base.DataMetadata().generate(value=outputs)

        for column_index, column_metadata in enumerate(target_columns_metadata):
            column_metadata.pop("structural_type", None)
            outputs_metadata = outputs_metadata.update_column(column_index, column_metadata)

        return outputs_metadata


    def _wrap_predictions(self, inputs: Inputs, predictions: ndarray) -> Outputs:
        outputs = d3m_dataframe(predictions, generate_metadata=False)
        outputs.metadata = self._update_predictions_metadata(inputs.metadata, outputs, self._target_columns_metadata)
        return outputs


    @classmethod
    def _add_target_columns_metadata(cls, outputs_metadata: metadata_base.DataMetadata):
        outputs_length = outputs_metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']

        target_columns_metadata: List[OrderedDict] = []
        for column_index in range(outputs_length):
            column_metadata = OrderedDict()
            semantic_types = []
            semantic_types.append('https://metadata.datadrivendiscovery.org/types/PredictedTarget')
            column_name = outputs_metadata.query((metadata_base.ALL_ELEMENTS, column_index)).get("name")
            if column_name is None:
                column_name = "output_{}".format(column_index)
            column_metadata["semantic_types"] = semantic_types
            column_metadata["name"] = str(column_name)
            target_columns_metadata.append(column_metadata)

        return target_columns_metadata
