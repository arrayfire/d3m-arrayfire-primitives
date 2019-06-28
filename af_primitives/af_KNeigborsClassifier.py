#! /usr/bin/python3

from time import time
from typing import Any, Callable, List, Dict, Union, Optional, Sequence, Tuple
import typing
from collections import OrderedDict
from numpy import ndarray
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

import arrayfire as af

# D3M interfaces
from d3m import utils
import common_primitives.utils as common_utils
from d3m.container.numpy import ndarray as d3m_ndarray
from d3m.container import DataFrame as d3m_dataframe
from d3m.metadata import hyperparams, params, base as metadata_base
from d3m.primitive_interfaces.base import CallResult, DockerContainer
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.base import ProbabilisticCompositionalityMixin

Inputs = d3m_dataframe
Outputs = d3m_dataframe

#TODO: figure out relevant params
class Params(params.Params):
    _fit_method: Optional[str]
    _fit_X: Optional[ndarray]
    _tree: Optional[object]
    classes_: Optional[ndarray]
    _y: Optional[ndarray]
    outputs_2d_: Optional[bool]
    effective_metric_: Optional[str]
    effective_metric_params_: Optional[Dict]
    radius: Optional[float]

    target_names_: Optional[Sequence[Any]]
    training_indices_: Optional[Sequence[int]]

class Hyperparams(hyperparams.Hyperparams):
    n_neighbors = hyperparams.Bounded[int](
        default=5,
        lower=1,
        upper=256,
        description='Number of neighbors to use by default for :meth:`k_neighbors` queries.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    weights = hyperparams.Enumeration[str](
        values=['uniform', 'distance'],
        default='uniform',
        description='weight function used in prediction.  Possible values:  - \'uniform\' : uniform weights. \
                     All points in each neighborhood are weighted equally. - \'distance\' : weight points by \
                     the inverse of their distance. in this case, closer neighbors of a query point will     \
                     have a greater influence than neighbors which are further away.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    metric = hyperparams.Enumeration[str](
        values=['euclidean', 'manhattan'],
        default='euclidean',
        description='the distance metric to use for the tree. The default metric is the standard Euclidean metric.',
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


class af_KNeighborsClassifier(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams],
                              ProbabilisticCompositionalityMixin[Inputs, Outputs, Params, Hyperparams]):
    """
    Primitive implementing KNeighborsClassifier using ArrayFire library
    """

    __author__ = "ArrayFire"
    metadata = metadata_base.PrimitiveMetadata({
        "algorithm_types": [metadata_base.PrimitiveAlgorithmType.K_NEAREST_NEIGHBORS, ],
        "id": "78c4acd6-ca23-456c-ab1c-c6d687b0957f",
        "name": "ArrayFire KNN Classifier",
        "python_path": "d3m.primitives.classification.k_neighbors.AF",
        "primitive_family": metadata_base.PrimitiveFamily.CLASSIFICATION,
        "version": "0.1.0"
        #"installation": [
            #{'type': metadata_base.PrimitiveInstallationType.PIP,
             #'package_uri': 'git+https://github.com/arrayfire/arrayfire-python'
            #TODO: git tag and egg?
            #}],
    })


    def __init__(self, *,
                 hyperparams: Hyperparams,
                 random_seed: int = 0,
                 docker_containers: Dict[str, DockerContainer] = None) -> None:

        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

        self.n_neighbors=self.hyperparams['n_neighbors'],
        self.weights=self.hyperparams['weights'],
        self.metric=self.hyperparams['metric'],

        self._training_inputs = None
        self._training_outputs = None
        self._target_names = None
        self._training_indices = None
        self._fitted = False

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        self._training_inputs, self._training_indices = self._get_columns_to_fit(inputs, self.hyperparams)
        self._training_outputs, self._target_names = self._get_targets(outputs, self.hyperparams)
        self._fitted = False

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        if self._fitted:
            return CallResult(None)

        if self._training_inputs is None or self._training_outputs is None:
            raise ValueError("Missing training data.")
        training_output = self._training_outputs.values

        shape = training_output.shape
        if len(shape) == 2 and shape[1] == 1:
            training_output = np.ravel(training_output)

        # "fit" data
        self._data = af.np_to_af_array(self._training_inputs)
        self._labels = af.np_to_af_array(training_outputs)

        self._fitted = True

        return CallResult(None)


    def _af_knn_predict(self, query):
        af_query = af.np_to_af_array(query)

        kids, kdists = af.vision.nearest_neighbour(af_query, self.data, dim=1, num_nearest=self.n, match_type=af.MATCH.SSD)
        labels = af.moddims(self.labels[kids], kids.shape[0], kids.shape[1])
        # TODO for k-NN:
        # for each unique in setUnique:
        #    idxs of unique in kids
        #    weight += kdists * "self.weights"
        # outlabel = argmax weight
        return af.flat(labels[0, :])

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        af_inputs = inputs
        if self.hyperparams['use_semantic_types']:
            af_inputs = inputs.iloc[:, self._training_indices]

        if sparse.issparse(af_inputs):
            af_inputs = af_inputs.toarray()

        af_output = self._af_knn_predict(sk_inputs)

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
                _fit_method=None,
                _fit_X=None,
                _tree=None,
                classes_=None,
                _y=None,
                outputs_2d_=None,
                effective_metric_=None,
                effective_metric_params_=None,
                radius=None,
                training_indices_=self._training_indices,
                target_names_=self._target_names
            )

        return Params(
            _fit_method=getattr(self._clf, '_fit_method', None),
            _fit_X=getattr(self._clf, '_fit_X', None),
            _tree=getattr(self._clf, '_tree', None),
            classes_=getattr(self._clf, 'classes_', None),
            _y=getattr(self._clf, '_y', None),
            outputs_2d_=getattr(self._clf, 'outputs_2d_', None),
            effective_metric_=getattr(self._clf, 'effective_metric_', None),
            effective_metric_params_=getattr(self._clf, 'effective_metric_params_', None),
            radius=getattr(self._clf, 'radius', None),
            training_indices_=self._training_indices,
            target_names_=self._target_names
        )

    def set_params(self, *, params: Params) -> None:
        self._clf._fit_method = params['_fit_method']
        self._clf._fit_X = params['_fit_X']
        self._clf._tree = params['_tree']
        self._clf.classes_ = params['classes_']
        self._clf._y = params['_y']
        self._clf.outputs_2d_ = params['outputs_2d_']
        self._clf.effective_metric_ = params['effective_metric_']
        self._clf.effective_metric_params_ = params['effective_metric_params_']
        self._clf.radius = params['radius']
        self._training_indices = params['training_indices_']
        self._target_names = params['target_names_']
        self._fitted = False

        if params['_fit_method'] is not None:
            self._fitted = True
        if params['_fit_X'] is not None:
            self._fitted = True
        if params['_tree'] is not None:
            self._fitted = True
        if params['classes_'] is not None:
            self._fitted = True
        if params['_y'] is not None:
            self._fitted = True
        if params['outputs_2d_'] is not None:
            self._fitted = True
        if params['effective_metric_'] is not None:
            self._fitted = True
        if params['effective_metric_params_'] is not None:
            self._fitted = True
        if params['radius'] is not None:
            self._fitted = True

    def log_likelihoods(self, *,
                    outputs: Outputs,
                    inputs: Inputs,
                    timeout: float = None,
                    iterations: int = None) -> CallResult[Sequence[float]]:

        inputs = inputs.values
        outputs = outputs.values
        # TODO: implement probs, nvotes / tot_votes
        # return CallResult(numpy.log(self._clf.predict_proba(inputs)[:, outputs]))
        return CallResult(None)

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
