from sklearn.impute import SimpleImputer as _SimpleImputer
from sklearn.impute._base import _most_frequent, _BaseImputer
from sklearn.utils.validation import (
        #ret = a @ b
    _deprecate_positional_args, _ensure_no_complex_data, _ensure_sparse_format, check_is_fitted, FLOAT_DTYPES)
from sklearn import get_config as _get_config
from sklearn.exceptions import DataConversionWarning
from sklearn.utils.fixes import _object_dtype_isnan  # FIXME
from sklearn.utils.sparsefuncs import _get_median
from afBaseEstimator import afBaseEstimator
from af_type_utils import typemap
from scipy import stats

import numbers
import warnings
import numpy.ma as ma
from numpy.core.numeric import ComplexWarning

import scipy.sparse as sp
from contextlib import suppress
import arrayfire as af
import numpy as np

def _most_frequent(array, extra_value, n_repeat):
    """Compute the most frequent value in a 1d array extended with
       [extra_value] * n_repeat, where extra_value is assumed to be not part
       of the array."""
    #TODO: af
    # Compute the most frequent value in array only
    if array.size > 0:
        with warnings.catch_warnings():
            # stats.mode raises a warning when input array contains objects due
            # to incapacity to detect NaNs. Irrelevant here since input array
            # has already been NaN-masked.
            warnings.simplefilter("ignore", RuntimeWarning)
            mode = stats.mode(array)

        most_frequent_value = mode[0][0]
        most_frequent_count = mode[1][0]
    else:
        most_frequent_value = 0
        most_frequent_count = 0

    # Compare to array + [extra_value] * n_repeat
    if most_frequent_count == 0 and n_repeat == 0:
        return np.nan
    elif most_frequent_count < n_repeat:
        return extra_value
    elif most_frequent_count > n_repeat:
        return most_frequent_value
    elif most_frequent_count == n_repeat:
        # Ties the breaks. Copy the behaviour of scipy.stats.mode
        if most_frequent_value < extra_value:
            return most_frequent_value
        else:
            return extra_value

class BaseImputer(_BaseImputer, afBaseEstimator):
    def _concatenate_indicator(self, X_imputed, X_indicator):
        """Concatenate indicator mask with the imputed data."""
        if not self.add_indicator:
            return X_imputed

        hstack = sp.hstack if sp.issparse(X_imputed) else af.hstack
        if X_indicator is not None:
            return hstack((X_imputed, X_indicator))

        raise ValueError(
            "Data from the missing indicator are not provided. Call "
            "_fit_indicator and _transform_indicator in the imputer implementation.")


class afSimpleImputer(_SimpleImputer, BaseImputer):

    def _validate_input(self, X, in_fit):  # NOTE: is duplicated due to a type checks
        #import pdb; pdb.set_trace()
        allowed_strategies = ["mean", "median", "most_frequent", "constant"]
        if self.strategy not in allowed_strategies:
            raise ValueError(f"Can only use these strategies: {allowed_strategies} got strategy={self.strategy}")

        if self.strategy in ("most_frequent", "constant"):
            dtype = None
        else:
            dtype = FLOAT_DTYPES

        if not is_scalar_nan(self.missing_values):
            force_all_finite = True
        else:
            force_all_finite = "allow-nan"

        try:
            X = self._validate_data(
                X, reset=in_fit, accept_sparse='csc', dtype=dtype, force_all_finite=force_all_finite, copy=self.copy)
        except ValueError as ve:
            if "could not convert" in str(ve):
                new_ve = ValueError("Cannot use {self.strategy} strategy with non-numeric data:\n{ve}")
                raise new_ve from None
            else:
                raise ve

        # BUG
        # _check_inputs_dtype(X, self.missing_values)
        # if X.dtype.kind not in ("i", "u", "f", "O"):
        #     raise ValueError("SimpleImputer does not support data with dtype "
        #                      f"{X.dtype}. Please provide either a numeric array (with"
        #                      " a floating point or integer dtype) or "
        #                      "categorical data represented either as an array "
        #                      "with integer dtype or an array of string values "
        #                      "with an object dtype.")

        return X

    def fit(self, X, y=None):
        """Fit the imputer on X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.
        Returns
        -------
        self : SimpleImputer
        """
        X = self._validate_input(X, in_fit=True)
        super()._fit_indicator(X)

        # default fill_value is 0 for numerical input and "missing_value"
        # otherwise
        # BUG uncomment below
        # if self.fill_value is None:
        #     if X.dtype.kind in ("i", "u", "f"):
        #         fill_value = 0
        #     else:
        #         fill_value = "missing_value"
        # else:
        #     fill_value = self.fill_value
        fill_value = 0  # FIXME: remove after bug is fixed

        # fill_value should be numerical in case of numerical input
        npdtype = typemap(X.dtype())
        if self.strategy == "constant" and npdtype.kind in ("i", "u", "f") and not isinstance(fill_value, numbers.Real):
            raise ValueError(
                f"'fill_value'={fill_value} is invalid. Expected a numerical value when imputing numerical data")

        if sp.issparse(X):
            # missing_values = 0 not allowed with sparse data as it would
            # force densification
            if self.missing_values == 0:
                raise ValueError(
                    "Imputation not possible when missing_values "
                    "== 0 and input is sparse. Provide a dense array instead.")
            else:
                self.statistics_ = self._sparse_fit(X, self.strategy, self.missing_values, fill_value)
        else:
            self.statistics_ = self._dense_fit(X, self.strategy, self.missing_values, fill_value)
        return self

    def _sparse_fit(self, X, strategy, missing_values, fill_value):
        """Fit the transformer on sparse data."""
        mask_data = _get_mask(X.data, missing_values)
        n_implicit_zeros = X.shape[0] - af.diff(X.indptr)

        statistics = af.empty(X.shape[1])

        if strategy == "constant":
            # for constant strategy, self.statistcs_ is used to store
            # fill_value in each column
            statistics.fill(fill_value)
        else:
            for i in range(X.shape[1]):
                column = X.data[X.indptr[i]:X.indptr[i + 1]]
                mask_column = mask_data[X.indptr[i]:X.indptr[i + 1]]
                column = column[~mask_column]

                # combine explicit and implicit zeros
                mask_zeros = _get_mask(column, 0)
                column = column[~mask_zeros]
                n_explicit_zeros = mask_zeros.sum()
                n_zeros = n_implicit_zeros[i] + n_explicit_zeros

                if strategy == "mean":
                    s = column.size + n_zeros
                    statistics[i] = np.nan if s == 0 else column.sum() / s

                elif strategy == "median":
                    statistics[i] = _get_median(column, n_zeros)

                elif strategy == "most_frequent":
                    statistics[i] = _most_frequent(column, 0, n_zeros)
        return statistics

    def _dense_fit(self, X, strategy, missing_values, fill_value):
        """Fit the transformer on dense data."""
        mask = _get_mask(X, missing_values)
        X_np = X.to_ndarray()
        masked_X = ma.masked_array(X_np, mask=np.isnan(X_np))  # FIXME

        # Mean
        if strategy == "mean":
            mean_masked = ma.mean(masked_X, axis=0)
            # Avoid the warning "Warning: converting a masked element to nan."
            mean = ma.getdata(mean_masked)
            mean[ma.getmask(mean_masked)] = np.nan

            return mean

        # Median
        elif strategy == "median":
            median_masked = ma.median(masked_X, axis=0)
            # Avoid the warning "Warning: converting a masked element to nan."
            median = ma.getdata(median_masked)
            median[ma.getmaskarray(median_masked)] = np.nan

            return median

        # Most frequent
        elif strategy == "most_frequent":
            # Avoid use of scipy.stats.mstats.mode due to the required
            # additional overhead and slow benchmarking performance.
            # See Issue 14325 and PR 14399 for full discussion.

            # To be able access the elements by columns

            X = X.T
            mask = mask.T

            npdtype = typemap(X.dtype())
            if npdtype.kind == "O":
                most_frequent = af.constant(0, X.shape[0], dtype=object)
            else:
                most_frequent = af.constant(0, X.shape[0])

            for i, (row, row_mask) in enumerate(zip(X[:], mask[:])):
                row_mask = row_mask.logical_not()
                row = row[row_mask].to_ndarray()
                #most_frequent[i] = _most_frequent(row, np.nan, 0)
                most_frequent[i] = _most_frequent(row, np.nan, 0)

            return most_frequent

        # Constant
        elif strategy == "constant":
            # for constant strategy, self.statistcs_ is used to store
            # fill_value in each column
            return af.constant(fill_value, X.shape[1], dtype=X.dtype())

    def transform(self, X):
        """Impute all missing values in X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data to complete.
        """
        check_is_fitted(self)

        X = self._validate_input(X, in_fit=False)
        #X = af.Array.to_ndarray(X)
        X_indicator = super()._transform_indicator(X)

        statistics = self.statistics_

        if X.shape[1] != statistics.shape[0]:
            raise ValueError(f"X has {X.shape[1]} features per sample, expected {self.statistics_.shape[0]}")

        # Delete the invalid columns if strategy is not constant
        if self.strategy == "constant":
            valid_statistics = statistics
        else:
            # same as af.isnan but also works for object dtypes
            # invalid_mask = _get_mask(statistics, np.nan)  # BUG: af runtime error
            invalid_mask = af.isnan(statistics)  # FIXME
            valid_mask = invalid_mask.logical_not()
            valid_statistics = statistics[valid_mask]
            valid_statistics_indexes = np.flatnonzero(valid_mask)

            if af.any_true(invalid_mask):
                missing = af.arange(X.shape[1])[invalid_mask]
                if self.verbose:
                    warnings.warn(f"Deleting features without observed values: {missing}")
                X = X[:, valid_statistics_indexes]

        # Do actual imputation
        if sp.issparse(X):
            if self.missing_values == 0:
                raise ValueError(
                    "Imputation not possible when missing_values == 0 and input is sparse."
                    "Provide a dense array instead.")
            else:
                mask = _get_mask(X.data, self.missing_values)
                indexes = af.repeat(af.arange(len(X.indptr) - 1, dtype=af.int), af.diff(X.indptr))[mask]

                X.data[mask] = valid_statistics[indexes].astype(X.dtype, copy=False)
        else:
            # mask = _get_mask(X, self.missing_values)  # BUG
            mask = af.isnan(X)  # FIXME
            # n_missing = af.sum(mask, axis=0)  # BUG af
            n_missing = af.sum(mask, dim=0)
            coordinates = af.where(mask.T)[::-1]  # BUG
            valid_statistics = valid_statistics.to_ndarray().ravel()
            n_missing = n_missing.to_ndarray().ravel()
            values = np.repeat(valid_statistics, n_missing)  # BUG
            values = af.interop.from_ndarray(values)

            odims = X.dims()
            X = af.flat(X)
            X[coordinates] = values
            X = af.moddims(X, *odims)

        return super()._concatenate_indicator(X, X_indicator)

def _get_mask(X, value_to_mask):
    """Compute the boolean mask X == value_to_mask."""
    # BUG: doesnt work properly
    npdtype = typemap(X.dtype())
    if is_scalar_nan(value_to_mask):
        if npdtype.kind == "f":
            return af.isnan(X)
        elif X.dtype.kind in ("i", "u"):
            # can't have NaNs in integer array.
            return af.constant(0, X.shape[0], X.shape[1], dtype=af.Dtype.b8)
        else:
            # np.isnan does not work on object dtypes.
            return _object_dtype_isnan(X) #todo:fix
    else:
        return X == value_to_mask


def is_scalar_nan(x):
    """
    Ref: https://github.com/scikit-learn/scikit-learn/blob/fd237278e895b42abe8d8d09105cbb82dc2cbba7/sklearn/utils/__init__.py#L1004
    """
    # convert from numpy.bool_ to python bool to ensure that testing
    # is_scalar_nan(x) is True does not fail.
    # import ipdb; ipdb.set_trace()
    return bool(isinstance(x, numbers.Real) and np.isnan(x))


def _check_inputs_dtype(X, missing_values):
    if X.dtype.kind in ("f", "i", "u") and not isinstance(missing_values, numbers.Real):
        raise ValueError(
            "'X' and 'missing_values' types are expected to be"
            f" both numerical. Got X.dtype={X.dtype} and type(missing_values)={type(missing_values)}.")
