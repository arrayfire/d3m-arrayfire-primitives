import arrayfire as af
#import cupy as np
import numpy as np
import numpy
import scipy.sparse as sp
import warnings
import numbers
from collections.abc import Sequence
from scipy.sparse.base import spmatrix
from itertools import chain

from sklearn.utils.validation import _deprecate_positional_args
from sklearn.preprocessing import LabelBinarizer
from af_validation import _num_samples
from af_validation import check_is_fitted
from af_validation import check_array
from af_validation import column_or_1d

def _unique_multiclass(y):
    if hasattr(y, '__array__'):
        return np.unique(np.asarray(y))
    else:
        return set(list(y))


def _unique_indicator(y):
    return np.arange(
        check_array(y, accept_sparse=['csr', 'csc', 'coo']).shape[1]
    )


_FN_UNIQUE_LABELS = {
    'binary': _unique_multiclass,
    'multiclass': _unique_multiclass,
    'multilabel-indicator': _unique_indicator,
}

def unique_labels(*ys):
    """Extract an ordered array of unique labels

    We don't allow:
        - mix of multilabel and multiclass (single label) targets
        - mix of label indicator matrix and anything else,
          because there are no explicit labels)
        - mix of label indicator matrices of different sizes
        - mix of string and integer labels

    At the moment, we also don't allow "multiclass-multioutput" input type.

    Parameters
    ----------
    *ys : array-likes

    Returns
    -------
    out : numpy array of shape [n_unique_labels]
        An ordered array of unique labels.

    Examples
    --------
    >>> from sklearn.utils.multiclass import unique_labels
    >>> unique_labels([3, 5, 5, 5, 7, 7])
    array([3, 5, 7])
    >>> unique_labels([1, 2, 3, 4], [2, 2, 3, 4])
    array([1, 2, 3, 4])
    >>> unique_labels([1, 2, 10], [5, 11])
    array([ 1,  2,  5, 10, 11])
    """
    if not ys:
        raise ValueError('No argument has been passed.')
    # Check that we don't mix label format

    ys_types = set(type_of_target(x) for x in ys)
    if ys_types == {"binary", "multiclass"}:
        ys_types = {"multiclass"}

    if len(ys_types) > 1:
        raise ValueError("Mix type of y not allowed, got types %s" % ys_types)

    label_type = ys_types.pop()

    # Check consistency for the indicator format
    if (label_type == "multilabel-indicator" and
            len(set(check_array(y,
                                accept_sparse=['csr', 'csc', 'coo']).shape[1]
                    for y in ys)) > 1):
        raise ValueError("Multi-label binary indicator input with "
                         "different numbers of labels")

    # Get the unique set of labels
    _unique_labels = _FN_UNIQUE_LABELS.get(label_type, None)
    if not _unique_labels:
        raise ValueError("Unknown label type: %s" % repr(ys))

    #ys_labels = set(chain.from_iterable(_unique_labels(y.tolist()) for y in ys))
    ys_labels = set(chain.from_iterable(_unique_labels(y.to_list()) for y in ys))

    # Check that we don't mix string type with number type
    if (len(set(isinstance(label, str) for label in ys_labels)) > 1):
        raise ValueError("Mix of label input types (string and number)")

    return np.array(sorted(ys_labels))

def is_multilabel(y):
    """ Check if ``y`` is in a multilabel format.

    Parameters
    ----------
    y : numpy array of shape [n_samples]
        Target values.

    Returns
    -------
    out : bool,
        Return ``True``, if ``y`` is in a multilabel format, else ```False``.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.utils.multiclass import is_multilabel
    >>> is_multilabel([0, 1, 0, 1])
    False
    >>> is_multilabel([[1], [0, 2], []])
    False
    >>> is_multilabel(np.array([[1, 0], [0, 0]]))
    True
    >>> is_multilabel(np.array([[1], [0], [0]]))
    False
    >>> is_multilabel(np.array([[1, 0, 0]]))
    True
    """
    if hasattr(y, '__array__') or isinstance(y, Sequence):
        y = np.asarray(y)
    if not (hasattr(y, "shape") and y.ndim == 2 and y.shape[1] > 1):
        return False

    if issparse(y):
        if isinstance(y, (dok_matrix, lil_matrix)):
            y = y.tocsr()
        return (len(y.data) == 0 or np.unique(y.data).size == 1 and
                (y.dtype.kind in 'biu' or  # bool, int, uint
                 _is_integral_float(np.unique(y.data))))
    else:
        labels = np.unique(y)

        return len(labels) < 3 and (y.dtype.kind in 'biu' or  # bool, int, uint
                                    _is_integral_float(labels))

def type_of_target(y):
    """Determine the type of data indicated by the target.

    Note that this type is the most specific type that can be inferred.
    For example:

        * ``binary`` is more specific but compatible with ``multiclass``.
        * ``multiclass`` of integers is more specific but compatible with
          ``continuous``.
        * ``multilabel-indicator`` is more specific but compatible with
          ``multiclass-multioutput``.

    Parameters
    ----------
    y : array-like

    Returns
    -------
    target_type : string
        One of:

        * 'continuous': `y` is an array-like of floats that are not all
          integers, and is 1d or a column vector.
        * 'continuous-multioutput': `y` is a 2d array of floats that are
          not all integers, and both dimensions are of size > 1.
        * 'binary': `y` contains <= 2 discrete values and is 1d or a column
          vector.
        * 'multiclass': `y` contains more than two discrete values, is not a
          sequence of sequences, and is 1d or a column vector.
        * 'multiclass-multioutput': `y` is a 2d array that contains more
          than two discrete values, is not a sequence of sequences, and both
          dimensions are of size > 1.
        * 'multilabel-indicator': `y` is a label indicator matrix, an array
          of two dimensions with at least two columns, and at most 2 unique
          values.
        * 'unknown': `y` is array-like but none of the above, such as a 3d
          array, sequence of sequences, or an array of non-sequence objects.

    Examples
    --------
    >>> import numpy as np
    >>> type_of_target([0.1, 0.6])
    'continuous'
    >>> type_of_target([1, -1, -1, 1])
    'binary'
    >>> type_of_target(['a', 'b', 'a'])
    'binary'
    >>> type_of_target([1.0, 2.0])
    'binary'
    >>> type_of_target([1, 0, 2])
    'multiclass'
    >>> type_of_target([1.0, 0.0, 3.0])
    'multiclass'
    >>> type_of_target(['a', 'b', 'c'])
    'multiclass'
    >>> type_of_target(np.array([[1, 2], [3, 1]]))
    'multiclass-multioutput'
    >>> type_of_target([[1, 2]])
    'multilabel-indicator'
    >>> type_of_target(np.array([[1.5, 2.0], [3.0, 1.6]]))
    'continuous-multioutput'
    >>> type_of_target(np.array([[0, 1], [1, 1]]))
    'multilabel-indicator'
    """
    valid = ((isinstance(y, (Sequence, spmatrix)) or hasattr(y, '__array__'))
             and not isinstance(y, str))

    if not valid:
        raise ValueError('Expected array-like (array or non-string sequence), '
                         'got %r' % y)

    sparse_pandas = (y.__class__.__name__ in ['SparseSeries', 'SparseArray'])
    if sparse_pandas:
        raise ValueError("y cannot be class 'SparseSeries' or 'SparseArray'")

    if is_multilabel(y):
        return 'multilabel-indicator'

    try:
        y = np.asarray(y)
    except ValueError:
        # Known to fail in numpy 1.3 for array of arrays
        return 'unknown'

    # The old sequence of sequences format
    try:
        if (not hasattr(y[0], '__array__') and isinstance(y[0], Sequence)
                and not isinstance(y[0], str)):
            raise ValueError('You appear to be using a legacy multi-label data'
                             ' representation. Sequence of sequences are no'
                             ' longer supported; use a binary array or sparse'
                             ' matrix instead - the MultiLabelBinarizer'
                             ' transformer can convert to this format.')
    except IndexError:
        pass

    # Invalid inputs
    if y.ndim > 2 or (y.dtype == object and len(y) and
                      not isinstance(y.flat[0], str)):
        return 'unknown'  # [[[1, 2]]] or [obj_1] and not ["label_1"]

    if y.ndim == 2 and y.shape[1] == 0:
        return 'unknown'  # [[]]

    if y.ndim == 2 and y.shape[1] > 1:
        suffix = "-multioutput"  # [[1, 2], [1, 2]]
    else:
        suffix = ""  # [1, 2, 3] or [[1], [2], [3]]

    # check float and contains non-integer float values
    if y.dtype.kind == 'f' and np.any(y != y.astype(int)):
        # [.1, .2, 3] or [[.1, .2, 3]] or [[1., .2]] and not [1., 2., 3.]
        _assert_all_finite(y)
        return 'continuous' + suffix

    if (len(np.unique(y)) > 2) or (y.ndim >= 2 and len(y[0]) > 1):
        return 'multiclass' + suffix  # [1, 2, 3] or [[1., 2., 3]] or [[1, 2]]
    else:
        return 'binary'  # [1, 2] or [["a"], ["b"]]

def _inverse_binarize_multiclass(y, classes):
    """Inverse label binarization transformation for multiclass.

    Multiclass uses the maximal score instead of a threshold.
    """
    classes = np.asarray(classes)

    if sp.issparse(y):
        # Find the argmax for each row in y where y is a CSR matrix

        y = y.tocsr()
        n_samples, n_outputs = y.shape
        outputs = np.arange(n_outputs)
        row_max = min_max_axis(y, 1)[1]
        row_nnz = np.diff(y.indptr)

        y_data_repeated_max = np.repeat(row_max, row_nnz)
        # picks out all indices obtaining the maximum per row
        y_i_all_argmax = np.flatnonzero(y_data_repeated_max == y.data)

        # For corner case where last row has a max of 0
        if row_max[-1] == 0:
            y_i_all_argmax = np.append(y_i_all_argmax, [len(y.data)])

        # Gets the index of the first argmax in each row from y_i_all_argmax
        index_first_argmax = np.searchsorted(y_i_all_argmax, y.indptr[:-1])
        # first argmax of each row
        y_ind_ext = np.append(y.indices, [0])
        y_i_argmax = y_ind_ext[y_i_all_argmax[index_first_argmax]]
        # Handle rows of all 0
        y_i_argmax[np.where(row_nnz == 0)[0]] = 0

        # Handles rows with max of 0 that contain negative numbers
        samples = np.arange(n_samples)[(row_nnz > 0) &
                                       (row_max.ravel() == 0)]
        for i in samples:
            ind = y.indices[y.indptr[i]:y.indptr[i + 1]]
            y_i_argmax[i] = classes[np.setdiff1d(outputs, ind)][0]

        return classes[y_i_argmax]
    else:
        return classes.take(y.argmax(axis=1), mode="clip")


def _inverse_binarize_thresholding(y, output_type, classes, threshold):
    """Inverse label binarization transformation using thresholding."""

    if output_type == "binary" and y.ndim == 2 and y.shape[1] > 2:
        raise ValueError("output_type='binary', but y.shape = {0}".
                         format(y.shape))

    if output_type != "binary" and y.shape[1] != len(classes):
        raise ValueError("The number of class is not equal to the number of "
                         "dimension of y.")

    classes = np.asarray(classes)

    # Perform thresholding
    if sp.issparse(y):
        if threshold > 0:
            if y.format not in ('csr', 'csc'):
                y = y.tocsr()
            y.data = np.array(y.data > threshold, dtype=np.int)
            y.eliminate_zeros()
        else:
            y = np.array(y.toarray() > threshold, dtype=np.int)
    else:
        y = np.array(y > threshold, dtype=np.int)

    # Inverse transform data
    if output_type == "binary":
        if sp.issparse(y):
            y = y.toarray()
        if y.ndim == 2 and y.shape[1] == 2:
            return classes[y[:, 1]]
        else:
            if len(classes) == 1:
                return np.repeat(classes[0], len(y))
            else:
                return classes[y.ravel()]

    elif output_type == "multilabel-indicator":
        return y

    else:
        raise ValueError("{0} format is not supported".format(output_type))


def af_in1d(arr0, arr1):
    #temporarily perform computation in numy, potentially change to arrayfire
    a0 = arr0.to_ndarray()
    #a1 = arr1.to_ndarray()
    isin = np.in1d(a0,  arr1)
    return isin

@_deprecate_positional_args
def label_binarize(y, *, classes, neg_label=0, pos_label=1,
                   sparse_output=False):
    """Binarize labels in a one-vs-all fashion

    Several regression and binary classification algorithms are
    available in scikit-learn. A simple way to extend these algorithms
    to the multi-class classification case is to use the so-called
    one-vs-all scheme.

    This function makes it possible to compute this transformation for a
    fixed set of class labels known ahead of time.

    Parameters
    ----------
    y : array-like
        Sequence of integer labels or multilabel data to encode.

    classes : array-like of shape [n_classes]
        Uniquely holds the label for each class.

    neg_label : int (default: 0)
        Value with which negative labels must be encoded.

    pos_label : int (default: 1)
        Value with which positive labels must be encoded.

    sparse_output : boolean (default: False),
        Set to true if output binary array is desired in CSR sparse format

    Returns
    -------
    Y : numpy array or CSR matrix of shape [n_samples, n_classes]
        Shape will be [n_samples, 1] for binary problems.

    Examples
    --------
    >>> from sklearn.preprocessing import label_binarize
    >>> label_binarize([1, 6], classes=[1, 2, 4, 6])
    array([[1, 0, 0, 0],
           [0, 0, 0, 1]])

    The class ordering is preserved:

    >>> label_binarize([1, 6], classes=[1, 6, 4, 2])
    array([[1, 0, 0, 0],
           [0, 1, 0, 0]])

    Binary targets transform to a column vector

    >>> label_binarize(['yes', 'no', 'no', 'yes'], classes=['no', 'yes'])
    array([[1],
           [0],
           [0],
           [1]])

    See also
    --------
    LabelBinarizer : class used to wrap the functionality of label_binarize and
        allow for fitting to classes independently of the transform operation
    """
    if not isinstance(y, list):
        # XXX Workaround that will be removed when list of list format is
        # dropped
        y = check_array(y, accept_sparse='csr', ensure_2d=False, dtype=None)
    else:
        if _num_samples(y) == 0:
            raise ValueError('y has 0 samples: %r' % y)
    if neg_label >= pos_label:
        raise ValueError("neg_label={0} must be strictly less than "
                         "pos_label={1}.".format(neg_label, pos_label))

    if (sparse_output and (pos_label == 0 or neg_label != 0)):
        raise ValueError("Sparse binarization is only supported with non "
                         "zero pos_label and zero neg_label, got "
                         "pos_label={0} and neg_label={1}"
                         "".format(pos_label, neg_label))

    # To account for pos_label == 0 in the dense case
    pos_switch = pos_label == 0
    if pos_switch:
        pos_label = -neg_label

    y_type = type_of_target(y)
    if 'multioutput' in y_type:
        raise ValueError("Multioutput target data is not supported with label "
                         "binarization")
    if y_type == 'unknown':
        raise ValueError("The type of target data is not known")

    n_samples = y.shape[0] if sp.issparse(y) else len(y)
    n_classes = len(classes)
    classes = np.asarray(classes)

    if y_type == "binary":
        if n_classes == 1:
            if sparse_output:
                return sp.csr_matrix((n_samples, 1), dtype=int)
            else:
                Y = np.zeros((len(y), 1), dtype=np.int)
                Y += neg_label
                return Y
        elif len(classes) >= 3:
            y_type = "multiclass"

    sorted_class = np.sort(classes)
    if y_type == "multilabel-indicator":
        y_n_classes = y.shape[1] if hasattr(y, 'shape') else len(y[0])
        if classes.size != y_n_classes:
            raise ValueError("classes {0} mismatch with the labels {1}"
                             " found in the data"
                             .format(classes, unique_labels(y)))

    if y_type in ("binary", "multiclass"):
        y = column_or_1d(y)

        # pick out the known labels from y
        y_in_classes = af_in1d(y, classes)
        y_in_classes = af.interop.from_ndarray(y_in_classes, copy=True)
        y[y_in_classes]
        y_seen = y[y_in_classes]
        y_seen = y_seen.to_ndarray()
        indices = np.searchsorted(sorted_class, y_seen)
        indptr = np.hstack((0, np.cumsum(y_in_classes)))

        data = np.empty_like(indices)
        data.fill(pos_label)
        Y = data

        #Y = sp.csr_matrix((data, indices, indptr),
                          #shape=(n_samples, n_classes))
    elif y_type == "multilabel-indicator":
        Y = sp.csr_matrix(y)
        if pos_label != 1:
            data = np.empty_like(Y.data)
            data.fill(pos_label)
            Y.data = data
    else:
        raise ValueError("%s target data is not supported with label "
                         "binarization" % y_type)

    if not sparse_output:
        #Y = Y.toarray() #TODO: test if ndarray, then cast if not
        Y = Y.astype(int, copy=False)

        if neg_label != 0:
            Y[Y == 0] = neg_label

        if pos_switch:
            Y[Y == pos_label] = 0
    else:
        Y.data = Y.data.astype(int, copy=False)

    # preserve label ordering
    if np.any(classes != sorted_class):
        indices = np.searchsorted(sorted_class, classes)
        Y = Y[:, indices]

    if y_type == "binary":
        if sparse_output:
            Y = Y.getcol(-1)
        else:
            Y = Y[:, -1].reshape((-1, 1))


    #return Y
    return Y


class afLabelBinarizer(LabelBinarizer):
    def fit(self, y):
        """Fit label binarizer

        Parameters
        ----------
        y : array of shape [n_samples,] or [n_samples, n_classes]
            Target values. The 2-d matrix should only contain 0 and 1,
            represents multilabel classification.

        Returns
        -------
        self : returns an instance of self.
        """
        self.y_type_ = type_of_target(y)
        if 'multioutput' in self.y_type_:
            raise ValueError("Multioutput target data is not supported with "
                             "label binarization")
        if _num_samples(y) == 0:
            raise ValueError('y has 0 samples: %r' % y)

        self.sparse_input_ = sp.issparse(y)
        self.classes_ = unique_labels(y)
        return self

    def transform(self, y):
        """Transform multi-class labels to binary labels

        The output of transform is sometimes referred to by some authors as
        the 1-of-K coding scheme.

        Parameters
        ----------
        y : array or sparse matrix of shape [n_samples,] or \
            [n_samples, n_classes]
            Target values. The 2-d matrix should only contain 0 and 1,
            represents multilabel classification. Sparse matrix can be
            CSR, CSC, COO, DOK, or LIL.

        Returns
        -------
        Y : numpy array or CSR matrix of shape [n_samples, n_classes]
            Shape will be [n_samples, 1] for binary problems.
        """
        check_is_fitted(self)

        y_is_multilabel = type_of_target(y).startswith('multilabel')
        if y_is_multilabel and not self.y_type_.startswith('multilabel'):
            raise ValueError("The object was not fitted with multilabel"
                             " input.")

        return label_binarize(y, classes=self.classes_,
                              pos_label=self.pos_label,
                              neg_label=self.neg_label,
                              sparse_output=self.sparse_output)

    def inverse_transform(self, Y, threshold=None):
        """Transform binary labels back to multi-class labels

        Parameters
        ----------
        Y : numpy array or sparse matrix with shape [n_samples, n_classes]
            Target values. All sparse matrices are converted to CSR before
            inverse transformation.

        threshold : float or None
            Threshold used in the binary and multi-label cases.

            Use 0 when ``Y`` contains the output of decision_function
            (classifier).
            Use 0.5 when ``Y`` contains the output of predict_proba.

            If None, the threshold is assumed to be half way between
            neg_label and pos_label.

        Returns
        -------
        y : numpy array or CSR matrix of shape [n_samples] Target values.

        Notes
        -----
        In the case when the binary labels are fractional
        (probabilistic), inverse_transform chooses the class with the
        greatest value. Typically, this allows to use the output of a
        linear model's decision_function method directly as the input
        of inverse_transform.
        """
        check_is_fitted(self)

        if threshold is None:
            threshold = (self.pos_label + self.neg_label) / 2.

        Y = Y.to_ndarray()
        if Y.ndim == 1:
            Y = Y[:, np.newaxis]
        if self.y_type_ == "multiclass":
            y_inv = _inverse_binarize_multiclass(Y, self.classes_)
        else:
            y_inv = _inverse_binarize_thresholding(Y, self.y_type_,
                                                   self.classes_, threshold)

        if self.sparse_input_:
            y_inv = sp.csr_matrix(y_inv)
        elif sp.issparse(y_inv):
            y_inv = y_inv.toarray()

        af_yinv = af.from_ndarray(y_inv)
        return af_yinv
