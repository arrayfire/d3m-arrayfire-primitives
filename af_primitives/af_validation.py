import arrayfire as af
import cupy as np
import numpy
import scipy.sparse as sp
import warnings
import numbers
from numpy.core.numeric import ComplexWarning
from inspect import signature, isclass, Parameter
from af_type_utils import typemap

from sklearn.utils.validation import _deprecate_positional_args
from sklearn._config import get_config as _get_config

def check_consistent_length(*arrays):
    """Check that all arrays have consistent first dimensions.

    Checks whether all objects in arrays have the same shape or length.

    Parameters
    ----------
    *arrays : list or tuple of input objects.
        Objects that will be checked for consistent length.
    """

    lengths = [_num_samples(X) for X in arrays if X is not None]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError("Found input variables with inconsistent numbers of"
                         " samples: %r" % [int(l) for l in lengths])

def _safe_accumulator_op(op, x, *args, **kwargs):
    """
    This function provides numpy accumulator functions with a float64 dtype
    when used on a floating point input. This prevents accumulator overflow on
    smaller floating point dtypes.
    Parameters
    ----------
    op : function
        A numpy accumulator function such as np.mean or np.sum
    x : numpy array
        A numpy array to apply the accumulator function
    *args : positional arguments
        Positional arguments passed to the accumulator function after the
        input x
    **kwargs : keyword arguments
        Keyword arguments passed to the accumulator function
    Returns
    -------
    result : The output of the accumulator function passed to this function
    """
    if np.issubdtype(x.dtype, np.floating) and x.dtype.itemsize < 8:
        result = op(x, *args, **kwargs, dtype=np.float64)
    else:
        result = op(x, *args, **kwargs)
    return result

def _determine_key_type(key, accept_slice=True):
    """Determine the data type of key.

    Parameters
    ----------
    key : scalar, slice or array-like
        The key from which we want to infer the data type.

    accept_slice : bool, default=True
        Whether or not to raise an error if the key is a slice.

    Returns
    -------
    dtype : {'int', 'str', 'bool', None}
        Returns the data type of key.
    """
    err_msg = ("No valid specification of the columns. Only a scalar, list or "
               "slice of all integers or all strings, or boolean mask is "
               "allowed")

    dtype_to_str = {int: 'int', str: 'str', bool: 'bool', np.bool_: 'bool'}
    array_dtype_to_str = {'i': 'int', 'u': 'int', 'b': 'bool', 'O': 'str',
                          'U': 'str', 'S': 'str'}

    if key is None:
        return None
    if isinstance(key, tuple(dtype_to_str.keys())):
        try:
            return dtype_to_str[type(key)]
        except KeyError:
            raise ValueError(err_msg)
    if isinstance(key, slice):
        if not accept_slice:
            raise TypeError(
                'Only array-like or scalar are supported. '
                'A Python slice was given.'
            )
        if key.start is None and key.stop is None:
            return None
        key_start_type = _determine_key_type(key.start)
        key_stop_type = _determine_key_type(key.stop)
        if key_start_type is not None and key_stop_type is not None:
            if key_start_type != key_stop_type:
                raise ValueError(err_msg)
        if key_start_type is not None:
            return key_start_type
        return key_stop_type
    if isinstance(key, (list, tuple)):
        unique_key = set(key)
        key_type = {_determine_key_type(elt) for elt in unique_key}
        if not key_type:
            return None
        if len(key_type) != 1:
            raise ValueError(err_msg)
        return key_type.pop()
    if hasattr(key, 'dtype'):
        try:
            return array_dtype_to_str[key.dtype.kind]
        except KeyError:
            raise ValueError(err_msg)
    raise ValueError(err_msg)

def _array_indexing(array, key, key_dtype, axis):
    """Index an array or scipy.sparse consistently across NumPy version."""
    if isinstance(key, tuple):
        key = list(key)
    afkey = af.interop.from_ndarray(key) #TODO: replace w/arrayfire keys
    return array[afkey, :] if axis == 0 else array[:, afkey]

def _safe_indexing(X, indices, *, axis=0):
    """Return rows, items or columns of X using indices.

    .. warning::

        This utility is documented, but **private**. This means that
        backward compatibility might be broken without any deprecation
        cycle.

    Parameters
    ----------
    X : array-like, sparse-matrix, list, pandas.DataFrame, pandas.Series
        Data from which to sample rows, items or columns. `list` are only
        supported when `axis=0`.
    indices : bool, int, str, slice, array-like
        - If `axis=0`, boolean and integer array-like, integer slice,
          and scalar integer are supported.
        - If `axis=1`:
            - to select a single column, `indices` can be of `int` type for
              all `X` types and `str` only for dataframe. The selected subset
              will be 1D, unless `X` is a sparse matrix in which case it will
              be 2D.
            - to select multiples columns, `indices` can be one of the
              following: `list`, `array`, `slice`. The type used in
              these containers can be one of the following: `int`, 'bool' and
              `str`. However, `str` is only supported when `X` is a dataframe.
              The selected subset will be 2D.
    axis : int, default=0
        The axis along which `X` will be subsampled. `axis=0` will select
        rows while `axis=1` will select columns.

    Returns
    -------
    subset
        Subset of X on axis 0 or 1.

    Notes
    -----
    CSR, CSC, and LIL sparse matrices are supported. COO sparse matrices are
    not supported.
    """
    if indices is None:
        return X

    if axis not in (0, 1):
        raise ValueError(
            "'axis' should be either 0 (to index rows) or 1 (to index "
            " column). Got {} instead.".format(axis)
        )

    indices_dtype = _determine_key_type(indices)

    if axis == 0 and indices_dtype == 'str':
        raise ValueError(
            "String indexing is not supported with 'axis=0'"
        )

    if axis == 1 and X.ndim != 2:
        raise ValueError(
            "'X' should be a 2D NumPy array, 2D sparse matrix or pandas "
            "dataframe when indexing the columns (i.e. 'axis=1'). "
            "Got {} instead with {} dimension(s).".format(type(X), X.ndim)
        )

    if axis == 1 and indices_dtype == 'str' and not hasattr(X, 'loc'):
        raise ValueError(
            "Specifying the columns using strings is only supported for "
            "pandas DataFrames"
        )

    if hasattr(X, "iloc"):
        return _pandas_indexing(X, indices, indices_dtype, axis=axis)
    elif hasattr(X, "shape"):
        return _array_indexing(X, indices, indices_dtype, axis=axis)
    else:
        return _list_indexing(X, indices, indices_dtype)

def _assert_all_finite(X, allow_nan=False, msg_dtype=None):
    """Like assert_all_finite, but only for ndarray."""
    # validation is also imported in extmath

    if _get_config()['assume_finite']:
        return
    X = np.asanyarray(X)
    # First try an O(n) time, O(1) space solution for the common case that
    # everything is finite; fall back to O(n) space np.isfinite to prevent
    # false positives from overflow in sum method. The sum is also calculated
    # safely to reduce dtype induced overflows.
    is_float = X.dtype.kind in 'fc'
    if is_float and (np.isfinite(_safe_accumulator_op(np.sum, X))):
        pass
    elif is_float:
        msg_err = "Input contains {} or a value too large for {!r}."
        if (allow_nan and np.isinf(X).any() or
                not allow_nan and not np.isfinite(X).all()):
            type_err = 'infinity' if allow_nan else 'NaN, infinity'
            raise ValueError(
                    msg_err.format
                    (type_err,
                     msg_dtype if msg_dtype is not None else X.dtype)
            )
    # for object dtype data, we only check for NaNs (GH-13254)
    elif X.dtype == np.dtype('object') and not allow_nan:
        if _object_dtype_isnan(X).any():
            raise ValueError("Input contains NaN")

def _ensure_no_complex_data(array):
    if hasattr(array, 'dtype') and array.dtype is not None \
            and hasattr(array.dtype, 'kind') and array.dtype.kind == "c":
        raise ValueError("Complex data not supported\n"
                         "{}\n".format(array))

def _num_samples(x):
    """Return number of samples in array-like x."""
    message = 'Expected sequence or array-like, got %s' % type(x)
    if hasattr(x, 'fit') and callable(x.fit):
        # Don't get num_samples from an ensembles length!
        raise TypeError(message)

    if not hasattr(x, '__len__') and not hasattr(x, 'shape'):
        if hasattr(x, '__array__'):
            x = np.asarray(x)
        else:
            raise TypeError(message)

    if hasattr(x, 'shape') and x.shape is not None:
        if len(x.shape) == 0:
            raise TypeError("Singleton array %r cannot be considered"
                            " a valid collection." % x)
        # Check that shape is returning an integer or default to len
        # Dask dataframes may not return numeric shape[0] value
        if isinstance(x.shape[0], numbers.Integral):
            return x.shape[0]

    try:
        return len(x)
    except TypeError:
        raise TypeError(message)


@_deprecate_positional_args
def check_array(array, accept_sparse=False, *, accept_large_sparse=True,
                dtype="numeric", order=None, copy=False, force_all_finite=True,
                ensure_2d=True, allow_nd=False, ensure_min_samples=1,
                ensure_min_features=1, estimator=None):

    """Input validation on an array, list, sparse matrix or similar.

    By default, the input is checked to be a non-empty 2D array containing
    only finite values. If the dtype of the array is object, attempt
    converting to float, raising on failure.

    Parameters
    ----------
    array : object
        Input object to check / convert.

    accept_sparse : string, boolean or list/tuple of strings (default=False)
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc. If the input is sparse but not in the allowed format,
        it will be converted to the first listed format. True allows the input
        to be any format. False means that a sparse matrix input will
        raise an error.

    accept_large_sparse : bool (default=True)
        If a CSR, CSC, COO or BSR sparse matrix is supplied and accepted by
        accept_sparse, accept_large_sparse=False will cause it to be accepted
        only if its indices are stored with a 32-bit dtype.

        .. versionadded:: 0.20

    dtype : string, type, list of types or None (default="numeric")
        Data type of result. If None, the dtype of the input is preserved.
        If "numeric", dtype is preserved unless array.dtype is object.
        If dtype is a list of types, conversion on the first type is only
        performed if the dtype of the input is not in the list.

    order : 'F', 'C' or None (default=None)
        Whether an array will be forced to be fortran or c-style.
        When order is None (default), then if copy=False, nothing is ensured
        about the memory layout of the output array; otherwise (copy=True)
        the memory layout of the returned array is kept as close as possible
        to the original array.

    copy : boolean (default=False)
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    force_all_finite : boolean or 'allow-nan', (default=True)
        Whether to raise an error on np.inf, np.nan, pd.NA in array. The
        possibilities are:

        - True: Force all values of array to be finite.
        - False: accepts np.inf, np.nan, pd.NA in array.
        - 'allow-nan': accepts only np.nan and pd.NA values in array. Values
          cannot be infinite.

        .. versionadded:: 0.20
           ``force_all_finite`` accepts the string ``'allow-nan'``.

        .. versionchanged:: 0.23
           Accepts `pd.NA` and converts it into `np.nan`

    ensure_2d : boolean (default=True)
        Whether to raise a value error if array is not 2D.

    allow_nd : boolean (default=False)
        Whether to allow array.ndim > 2.

    ensure_min_samples : int (default=1)
        Make sure that the array has a minimum number of samples in its first
        axis (rows for a 2D array). Setting to 0 disables this check.

    ensure_min_features : int (default=1)
        Make sure that the 2D array has some minimum number of features
        (columns). The default value of 1 rejects empty datasets.
        This check is only enforced when the input data has effectively 2
        dimensions or is originally 1D and ``ensure_2d`` is True. Setting to 0
        disables this check.

    estimator : str or estimator instance (default=None)
        If passed, include the name of the estimator in warning messages.

    Returns
    -------
    array_converted : object
        The converted and validated array.
    """
    # store reference to original array to check if copy is needed when
    # function returns
    array_orig = array
    #import pdb;pdb.set_trace()
    return array #TMP todo: perform checks for af::array

    # store whether originally we wanted numeric dtype
    dtype_numeric = isinstance(dtype, str) and dtype == "numeric"

    dtype_orig = getattr(array, "dtype", None)
    if not hasattr(dtype_orig, 'kind'):
        # not a data type (e.g. a column named dtype in a pandas DataFrame)
        dtype_orig = None

    # check if the object contains several dtypes (typically a pandas
    # DataFrame), and store them. If not, store None.
    dtypes_orig = None
    has_pd_integer_array = False
    if hasattr(array, "dtypes") and hasattr(array.dtypes, '__array__'):
        # throw warning if columns are sparse. If all columns are sparse, then
        # array.sparse exists and sparsity will be perserved (later).
        with suppress(ImportError):
            from pandas.api.types import is_sparse
            if (not hasattr(array, 'sparse') and
                    array.dtypes.apply(is_sparse).any()):
                warnings.warn(
                    "pandas.DataFrame with sparse columns found."
                    "It will be converted to a dense numpy array."
                )

        dtypes_orig = list(array.dtypes)
        # pandas boolean dtype __array__ interface coerces bools to objects
        for i, dtype_iter in enumerate(dtypes_orig):
            if dtype_iter.kind == 'b':
                dtypes_orig[i] = np.dtype(np.object)
            elif dtype_iter.name.startswith(("Int", "UInt")):
                # name looks like an Integer Extension Array, now check for
                # the dtype
                with suppress(ImportError):
                    from pandas import (Int8Dtype, Int16Dtype,
                                        Int32Dtype, Int64Dtype,
                                        UInt8Dtype, UInt16Dtype,
                                        UInt32Dtype, UInt64Dtype)
                    if isinstance(dtype_iter, (Int8Dtype, Int16Dtype,
                                               Int32Dtype, Int64Dtype,
                                               UInt8Dtype, UInt16Dtype,
                                               UInt32Dtype, UInt64Dtype)):
                        has_pd_integer_array = True

        if all(isinstance(dtype, np.dtype) for dtype in dtypes_orig):
            dtype_orig = np.result_type(*dtypes_orig)

    if dtype_numeric:
        if dtype_orig is not None and dtype_orig.kind == "O":
            # if input is object, convert to float.
            dtype = np.float64
        else:
            dtype = None

    if isinstance(dtype, (list, tuple)):
        if dtype_orig is not None and dtype_orig in dtype:
            # no dtype conversion required
            dtype = None
        else:
            # dtype conversion required. Let's select the first element of the
            # list of accepted types.
            dtype = dtype[0]

    if has_pd_integer_array:
        # If there are any pandas integer extension arrays,
        array = array.astype(dtype)

    if force_all_finite not in (True, False, 'allow-nan'):
        raise ValueError('force_all_finite should be a bool or "allow-nan"'
                         '. Got {!r} instead'.format(force_all_finite))

    if estimator is not None:
        if isinstance(estimator, str):
            estimator_name = estimator
        else:
            estimator_name = estimator.__class__.__name__
    else:
        estimator_name = "Estimator"
    context = " by %s" % estimator_name if estimator is not None else ""

    # When all dataframe columns are sparse, convert to a sparse array
    if hasattr(array, 'sparse') and array.ndim > 1:
        # DataFrame.sparse only supports `to_coo`
        array = array.sparse.to_coo()

    if sp.issparse(array):
        _ensure_no_complex_data(array)
        array = _ensure_sparse_format(array, accept_sparse=accept_sparse,
                                      dtype=dtype, copy=copy,
                                      force_all_finite=force_all_finite,
                                      accept_large_sparse=accept_large_sparse)
    else:
        # If np.array(..) gives ComplexWarning, then we convert the warning
        # to an error. This is needed because specifying a non complex
        # dtype to the function converts complex to real dtype,
        # thereby passing the test made in the lines following the scope
        # of warnings context manager.
        with warnings.catch_warnings():
            try:
                warnings.simplefilter('error', ComplexWarning)
                if dtype is not None and np.dtype(dtype).kind in 'iu':
                    # Conversion float -> int should not contain NaN or
                    # inf (numpy#14412). We cannot use casting='safe' because
                    # then conversion float -> int would be disallowed.
                    array = np.asarray(array, order=order)
                    if array.dtype.kind == 'f':
                        _assert_all_finite(array, allow_nan=False,
                                           msg_dtype=dtype)
                    array = array.astype(dtype, casting="unsafe", copy=False)
                else:
                    array = np.asarray(array, order=order, dtype=dtype)
            except ComplexWarning:
                raise ValueError("Complex data not supported\n"
                                 "{}\n".format(array))

        # It is possible that the np.array(..) gave no warning. This happens
        # when no dtype conversion happened, for example dtype = None. The
        # result is that np.array(..) produces an array of complex dtype
        # and we need to catch and raise exception for such cases.
        _ensure_no_complex_data(array)

        if ensure_2d:
            # If input is scalar raise error
            if array.ndim == 0:
                raise ValueError(
                    "Expected 2D array, got scalar array instead:\narray={}.\n"
                    "Reshape your data either using array.reshape(-1, 1) if "
                    "your data has a single feature or array.reshape(1, -1) "
                    "if it contains a single sample.".format(array))
            # If input is 1D raise error
            if array.ndim == 1:
                raise ValueError(
                    "Expected 2D array, got 1D array instead:\narray={}.\n"
                    "Reshape your data either using array.reshape(-1, 1) if "
                    "your data has a single feature or array.reshape(1, -1) "
                    "if it contains a single sample.".format(array))

        # in the future np.flexible dtypes will be handled like object dtypes
        #if dtype_numeric and np.issubdtype(array.dtype, np.flexible):
            #warnings.warn(
                #"Beginning in version 0.22, arrays of bytes/strings will be "
                #"converted to decimal numbers if dtype='numeric'. "
                #"It is recommended that you convert the array to "
                #"a float dtype before using it in scikit-learn, "
                #"for example by using "
                #"your_array = your_array.astype(np.float64).",
                #FutureWarning, stacklevel=2)

        # make sure we actually converted to numeric:
        if dtype_numeric and array.dtype.kind == "O":
            array = array.astype(np.float64)
        if not allow_nd and array.ndim >= 3:
            raise ValueError("Found array with dim %d. %s expected <= 2."
                             % (array.ndim, estimator_name))

        if force_all_finite:
            _assert_all_finite(array,
                               allow_nan=force_all_finite == 'allow-nan')

    if ensure_min_samples > 0:
        n_samples = _num_samples(array)
        if n_samples < ensure_min_samples:
            raise ValueError("Found array with %d sample(s) (shape=%s) while a"
                             " minimum of %d is required%s."
                             % (n_samples, array.shape, ensure_min_samples,
                                context))

    if ensure_min_features > 0 and array.ndim == 2:
        n_features = array.shape[1]
        if n_features < ensure_min_features:
            raise ValueError("Found array with %d feature(s) (shape=%s) while"
                             " a minimum of %d is required%s."
                             % (n_features, array.shape, ensure_min_features,
                                context))

    if copy and np.may_share_memory(array, array_orig):
        array = np.array(array, dtype=dtype, order=order)

    return array

@_deprecate_positional_args
def check_X_y(X, y, accept_sparse=False, *, accept_large_sparse=True,
              dtype="numeric", order=None, copy=False, force_all_finite=True,
              ensure_2d=True, allow_nd=False, multi_output=False,
              ensure_min_samples=1, ensure_min_features=1, y_numeric=False,
              estimator=None):
    """Input validation for standard estimators.

    Checks X and y for consistent length, enforces X to be 2D and y 1D. By
    default, X is checked to be non-empty and containing only finite values.
    Standard input checks are also applied to y, such as checking that y
    does not have np.nan or np.inf targets. For multi-label y, set
    multi_output=True to allow 2D and sparse y. If the dtype of X is
    object, attempt converting to float, raising on failure.

    Parameters
    ----------
    X : nd-array, list or sparse matrix
        Input data.

    y : nd-array, list or sparse matrix
        Labels.

    accept_sparse : string, boolean or list of string (default=False)
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc. If the input is sparse but not in the allowed format,
        it will be converted to the first listed format. True allows the input
        to be any format. False means that a sparse matrix input will
        raise an error.

    accept_large_sparse : bool (default=True)
        If a CSR, CSC, COO or BSR sparse matrix is supplied and accepted by
        accept_sparse, accept_large_sparse will cause it to be accepted only
        if its indices are stored with a 32-bit dtype.

        .. versionadded:: 0.20

    dtype : string, type, list of types or None (default="numeric")
        Data type of result. If None, the dtype of the input is preserved.
        If "numeric", dtype is preserved unless array.dtype is object.
        If dtype is a list of types, conversion on the first type is only
        performed if the dtype of the input is not in the list.

    order : 'F', 'C' or None (default=None)
        Whether an array will be forced to be fortran or c-style.

    copy : boolean (default=False)
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    force_all_finite : boolean or 'allow-nan', (default=True)
        Whether to raise an error on np.inf, np.nan, pd.NA in X. This parameter
        does not influence whether y can have np.inf, np.nan, pd.NA values.
        The possibilities are:

        - True: Force all values of X to be finite.
        - False: accepts np.inf, np.nan, pd.NA in X.
        - 'allow-nan': accepts only np.nan or pd.NA values in X. Values cannot
          be infinite.

        .. versionadded:: 0.20
           ``force_all_finite`` accepts the string ``'allow-nan'``.

        .. versionchanged:: 0.23
           Accepts `pd.NA` and converts it into `np.nan`

    ensure_2d : boolean (default=True)
        Whether to raise a value error if X is not 2D.

    allow_nd : boolean (default=False)
        Whether to allow X.ndim > 2.

    multi_output : boolean (default=False)
        Whether to allow 2D y (array or sparse matrix). If false, y will be
        validated as a vector. y cannot have np.nan or np.inf values if
        multi_output=True.

    ensure_min_samples : int (default=1)
        Make sure that X has a minimum number of samples in its first
        axis (rows for a 2D array).

    ensure_min_features : int (default=1)
        Make sure that the 2D array has some minimum number of features
        (columns). The default value of 1 rejects empty datasets.
        This check is only enforced when X has effectively 2 dimensions or
        is originally 1D and ``ensure_2d`` is True. Setting to 0 disables
        this check.

    y_numeric : boolean (default=False)
        Whether to ensure that y has a numeric type. If dtype of y is object,
        it is converted to float64. Should only be used for regression
        algorithms.

    estimator : str or estimator instance (default=None)
        If passed, include the name of the estimator in warning messages.

    Returns
    -------
    X_converted : object
        The converted and validated X.

    y_converted : object
        The converted and validated y.
    """
    #import pdb; pdb.set_trace()
    if y is None:
        raise ValueError("y cannot be None")

    X = check_array(X, accept_sparse=accept_sparse,
                    accept_large_sparse=accept_large_sparse,
                    dtype=dtype, order=order, copy=copy,
                    force_all_finite=force_all_finite,
                    ensure_2d=ensure_2d, allow_nd=allow_nd,
                    ensure_min_samples=ensure_min_samples,
                    ensure_min_features=ensure_min_features,
                    estimator=estimator)
    if multi_output:
        y = check_array(y, accept_sparse='csr', force_all_finite=True,
                        ensure_2d=False, dtype=None)
    else:
        y = column_or_1d(y, warn=True)
        _assert_all_finite(y)
    npdtype = typemap(y.dtype())
    if y_numeric and npdtype == 'O':
        y = y.astype(af.Dype.f64)

    check_consistent_length(X, y)

    return X, y


@_deprecate_positional_args
def column_or_1d(y, *, warn=False):
    """ Ravel column or 1d numpy array, else raises an error

    Parameters
    ----------
    y : array-like

    warn : boolean, default False
       To control display of warnings.

    Returns
    -------
    y : array

    """
    #y = np.asarray(y)
    #shape = np.shape(y)
    shape = y.dims()
    if len(shape) == 1:
        #return np.ravel(y)
        return af.flat(y)
    if len(shape) == 2 and shape[1] == 1:
        if warn:
            warnings.warn("A column-vector y was passed when a 1d array was"
                          " expected. Please change the shape of y to "
                          "(n_samples, ), for example using ravel().",
                          DataConversionWarning, stacklevel=2)
        #return np.ravel(y)
        return af.flat(y)

    raise ValueError(
        "y should be a 1d array, "
        "got an array of shape {} instead.".format(shape))

@_deprecate_positional_args
def check_is_fitted(estimator, attributes=None, *, msg=None, all_or_any=all):
    """Perform is_fitted validation for estimator.

    Checks if the estimator is fitted by verifying the presence of
    fitted attributes (ending with a trailing underscore) and otherwise
    raises a NotFittedError with the given message.

    This utility is meant to be used internally by estimators themselves,
    typically in their own predict / transform methods.

    Parameters
    ----------
    estimator : estimator instance.
        estimator instance for which the check is performed.

    attributes : str, list or tuple of str, default=None
        Attribute name(s) given as string or a list/tuple of strings
        Eg.: ``["coef_", "estimator_", ...], "coef_"``

        If `None`, `estimator` is considered fitted if there exist an
        attribute that ends with a underscore and does not start with double
        underscore.

    msg : string
        The default error message is, "This %(name)s instance is not fitted
        yet. Call 'fit' with appropriate arguments before using this
        estimator."

        For custom messages if "%(name)s" is present in the message string,
        it is substituted for the estimator name.

        Eg. : "Estimator, %(name)s, must be fitted before sparsifying".

    all_or_any : callable, {all, any}, default all
        Specify whether all or any of the given attributes must exist.

    Returns
    -------
    None

    Raises
    ------
    NotFittedError
        If the attributes are not found.
    """
    if isclass(estimator):
        raise TypeError("{} is a class, not an instance.".format(estimator))
    if msg is None:
        msg = ("This %(name)s instance is not fitted yet. Call 'fit' with "
               "appropriate arguments before using this estimator.")

    if not hasattr(estimator, 'fit'):
        raise TypeError("%s is not an estimator instance." % (estimator))

    if attributes is not None:
        if not isinstance(attributes, (list, tuple)):
            attributes = [attributes]
        attrs = all_or_any([hasattr(estimator, attr) for attr in attributes])
    else:
        attrs = [v for v in vars(estimator)
                 if v.endswith("_") and not v.startswith("__")]

    if not attrs:
        raise NotFittedError(msg % {'name': type(estimator).__name__})

def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)

