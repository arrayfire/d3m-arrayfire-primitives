from typing import Any, Callable, List, Dict, Union, Optional, Sequence, Tuple
from numpy import ndarray
from collections import OrderedDict
from scipy import sparse
import os
import numpy
import typing

# Custom import commands if any
import arrayfire as af

from d3m.container.numpy import ndarray as d3m_ndarray
from d3m.container import DataFrame as d3m_dataframe
from d3m.metadata import hyperparams, params, base as metadata_base
from d3m import utils
from d3m.base import utils as base_utils
from d3m.exceptions import PrimitiveNotFittedError
from d3m.primitive_interfaces.base import CallResult, DockerContainer

from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.base import ProbabilisticCompositionalityMixin, ContinueFitMixin
from d3m import exceptions
import pandas

Inputs = d3m_dataframe
Outputs = d3m_dataframe


class Params(params.Params):
    input_column_names: Optional[Any]
    target_names_: Optional[Sequence[Any]]
    training_indices_: Optional[Sequence[int]]
    target_column_indices_: Optional[Sequence[int]]
    target_columns_metadata_: Optional[List[OrderedDict]]


class Hyperparams(hyperparams.Hyperparams):
    n_neighbors = hyperparams.Bounded[int](
        default=5,
        lower=0,
        upper=None,
        description='Number of neighbors to use by default for :meth:`k_neighbors` queries.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    weights = hyperparams.Enumeration[str](
        values=['uniform', 'distance'],
        default='uniform',
        description='weight function used in prediction.  Possible values:  - \'uniform\' : uniform weights.  All points in each neighborhood are weighted equally. - \'distance\' : weight points by the inverse of their distance. in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away. - [callable] : a user-defined function which accepts an array of distances, and returns an array of the same shape containing the weights.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    dist_type = hyperparams.Enumeration[str](
        values=['sad', 'ssd', 'hamming'],
        default='ssd',
        description='The distance computation type. Currently \'sad\' (sum of absolute differences), \'ssd\' (sum of squared differences), and \'hamming\' (hamming distances) are supported.',
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

class af_KNeighborsClassifier(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams],
                              ProbabilisticCompositionalityMixin[Inputs, Outputs, Params, Hyperparams]):
    """
    Primitive implementing KNeighborsClassifier using ArrayFire library
    """

    __author__ = 'ArrayFire'
    metadata = metadata_base.PrimitiveMetadata({
        'name': 'ArrayFire KNN Classifier',
        'source': {
            'name': 'ArrayFire',
            'contact': 'mailto:support@arrayfire.com',
            'uris': ['https://github.com/arrayfire/d3m-arrayfire-primitives.git']},
        'id': '78c4acd6-ca23-456c-ab1c-c6d687b0957f',
        'version': '0.1.0',
        'python_path': 'd3m.primitives.classification.k_neighbors.ArrayFire',
        'keywords' : ['arrayfire', 'knearestneighbors', 'knn'],
        'installation': [
            {'type': metadata_base.PrimitiveInstallationType.PIP,
             'package_uri': 'git+https://github.com/arrayfire/d3m-arrayfire-primitives.git{git_commit}#egg=af_primitives'.format(
                 git_commit=utils.current_git_commit(os.path.dirname(__file__)),
             ),
            }],
        'algorithm_types': [metadata_base.PrimitiveAlgorithmType.K_NEAREST_NEIGHBORS, ],
        'primitive_family': metadata_base.PrimitiveFamily.CLASSIFICATION,
        'hyperparams_to_tune': ['n_neighbors', 'dist_type'],
    })

    def __init__(self, *,
                 hyperparams: Hyperparams,
                 random_seed: int = 0,
                 docker_containers: Dict[str, DockerContainer] = None) -> None:

        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

        self._n_neighbors=self.hyperparams['n_neighbors'],
        self._weights=self.hyperparams['weights'],
        self._data = None
        self._labels = None

        self._inputs = None
        self._outputs = None
        self._training_inputs = None
        self._training_outputs = None
        self._target_names = None
        self._training_indices = None
        self._target_column_indices = None
        self._target_columns_metadata: List[OrderedDict] = None
        self._input_column_names = None
        self._fitted = False
        self._new_training_data = False

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
            training_output = self._training_outputs.values

            shape = training_output.shape
            if len(shape) == 2 and shape[1] == 1:
                training_output = numpy.ravel(training_output)

            # "fit" data
            self._data = af.from_ndarray(self._training_inputs.values)
            self._labels = af.from_ndarray(training_output.astype('int32'))

            self._fitted = True
        else:
            if self.hyperparams['error_on_no_input']:
                raise RuntimeError("No input columns were selected")
            self.logger.warn("No input columns were selected")

        return CallResult(None)


    @classmethod
    def _get_neighbor_weights(self, dists, weight_by_dist, k):
        weights = None
        if weight_by_dist:
            inv_dists = 1./dists
            sum_inv_dists = af.sum(inv_dists)
            weights = inv_dists / sum_inv_dists
        else:
            weights = af.Array.copy(dists)
            weights[:] = 1./k
        return weights


    @classmethod
    def _get_dist_type(self, dist_type_str):
        dist_type = None
        if dist_type_str == 'sad':
            dist_type = af.MATCH.SAD
        elif dist_type_str == 'ssd':
            dist_type = af.MATCH.SSD
        elif dist_type_str == 'hamming':
            dist_type = af.MATCH.SHD
        else:
            raise RuntimeError('Invalid ArrayFire nearest neighbour distance type')
        return dist_type


    @classmethod
    def _predict(self, query, train_feats, train_labels, k, dist_type, weight_by_dist):
        near_locs, near_dists = af.vision.nearest_neighbour(query, train_feats, 1, \
                                                            k, dist_type)
        weights = self._get_neighbor_weights(near_dists, weight_by_dist, k)
        top_labels = af.moddims(train_labels[near_locs], \
                                near_locs.dims()[0], near_locs.dims()[1])
        accum_weights = af.scan_by_key(top_labels, weights) # reduce by key would be more ideal
        _, max_weight_locs = af.imax(accum_weights, dim=0)
        pred_idxs = af.range(accum_weights.dims()[1]) * accum_weights.dims()[0] + max_weight_locs.T
        top_labels_flat = af.flat(top_labels)
        pred_classes = top_labels_flat[pred_idxs]
        return pred_classes


    @classmethod
    def _predict_proba(self, query, train_feats, train_labels, k, dist_type, weight_by_dist):
        near_locs, near_dists = af.vision.nearest_neighbour(query, train_feats, 1, \
                                                            k, dist_type)
        weights = self._get_neighbor_weights(near_dists, weight_by_dist, k)
        top_labels = af.moddims(train_labels[near_locs], \
                                near_locs.dims()[0], near_locs.dims()[1])
        accum_weights = af.scan_by_key(top_labels, weights) # reduce by key would be more ideal
        probs, _ = af.imax(accum_weights, dim=0)
        return probs.T


    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        sk_inputs, columns_to_use = self._get_columns_to_fit(inputs, self.hyperparams)
        output = []
        if len(sk_inputs.columns):
            try:
                af_inputs = af.from_ndarray(sk_inputs.values)
                weight_by_dist = self._weights == 'distance'
                dist_type = self._get_dist_type(self.hyperparams['dist_type'])
                af_output = self._predict(af_inputs, self._data, self._labels,        \
                                          self.hyperparams['n_neighbors'], dist_type, \
                                          weight_by_dist)
                af_ndarray_output = af_output.to_ndarray().astype('int32')
            except sklearn.exceptions.NotFittedError as error:
                raise PrimitiveNotFittedError("Primitive not fitted.") from error
            # For primitives that allow predicting without fitting like GaussianProcessRegressor
            if not self._fitted:
                raise PrimitiveNotFittedError("Primitive not fitted.")
            if sparse.issparse(af_ndarray_output):
                af_ndarray_output = af_ndarray_output.toarray()
            output = self._wrap_predictions(inputs, af_ndarray_output)
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
                input_column_names=self._input_column_names,
                training_indices_=self._training_indices,
                target_names_=self._target_names,
                target_column_indices_=self._target_column_indices,
                target_columns_metadata_=self._target_columns_metadata
            )

        return Params(
            input_column_names=self._input_column_names,
            training_indices_=self._training_indices,
            target_names_=self._target_names,
            target_column_indices_=self._target_column_indices,
            target_columns_metadata_=self._target_columns_metadata
        )


    def set_params(self, *, params: Params) -> None:
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
        sk_inputs, columns_to_use = self._get_columns_to_fit(inputs, self.hyperparams)
        af_inputs = af.from_ndarray(sk_inputs.values)
        weight_by_dist = self._weights == 'distance'
        dist_type = self._get_dist_type(self.hyperparams['dist_type'])
        probs = self._predict_proba(af_inputs, self._data, self._labels,        \
                                    self.hyperparams['n_neighbors'], dist_type, \
                                    weight_by_dist)
        return CallResult(af.log(probs).to_ndarray())


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

