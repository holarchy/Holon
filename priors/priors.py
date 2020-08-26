import pandas as pd
from math import floor, ceil
import common


def make_prior_from_df(df: pd.DataFrame):
    """

    :param my_perspective_machine_df:
    :return: dictionary
    """
    jsonable_dict = dict()

    prior_length = len(df)
    for var in common.numeric_vars():
        temp_dict = dict()
        temp_dict['nan_length'] = len(df[df[var].isnull()])
        temp_dict['prior_length'] = prior_length
        temp_series = df[df[var].notnull()][var]
        temp_dict['mean'] = temp_series.mean()
        temp_dict['stdev'] = temp_series.std()
        cdf_dict = dict()
        temp_series = temp_series.sort_values(ascending=True).reset_index()
        if len(temp_series) > 0:
            for val in [0.0, 2**-16, 2**-8, 2**-6, 2**-4, 2**-2, 2**-1]:
                cdf_dict[val] = temp_series.iloc[floor(val * len(temp_series))][var]
            for val in [1 - 2**-2, 1 - 2**-4, 1 - 2**-6, 1 - 2**-8, 1 - 2**-16, 1.0]:
                cdf_dict[val] = temp_series.iloc[ceil(val * (len(temp_series) - 1))][var]
        temp_dict['cdf'] = cdf_dict
        jsonable_dict[var] = temp_dict

    for var in common.binary_vars():
        temp_dict = dict()
        temp_dict['nan_length'] = len(df[df[var].isnull()])
        temp_dict['prior_length'] = prior_length
        temp_series = df[df[var].notnull()][var]
        temp_dict['num_unique'] = len(temp_series.unique())
        value_counts = temp_series.value_counts()

        cdf_dict = value_counts.to_dict()
        temp_dict['cdf'] = cdf_dict
        temp_dict['binary_mean'] = get_binary_mean(cdf_dict)
        jsonable_dict[var] = temp_dict

    for var in common.categorical_vars():
        temp_dict = dict()
        temp_dict['nan_length'] = len(df[df[var].isnull()])
        temp_dict['prior_length'] = prior_length
        temp_series = df[df[var].notnull()][var]
        temp_dict['num_unique'] = len(temp_series.unique())
        value_counts = temp_series.value_counts()

        cdf_dict = value_counts.to_dict()
        temp_dict['cdf'] = cdf_dict

        # category equivalent stdev
        stdev_scaler = 1
        values = cdf_dict.values()
        if len(cdf_dict) == 0:
            # raise ValueError(f'There is nothing in the variable\'s CDF')
            distance = 1
            category_equivalent_stdev = 1
        elif len(cdf_dict) == 1:  # todo review this. looking out for variance error
            distance = 1
            category_equivalent_stdev = 1
        else:
            distance = (1 - (max(values) / sum(values))) ** stdev_scaler
            category_equivalent_stdev = fast_stdev(cdf_dict, distance)

        temp_dict['stdev_scaler'] = stdev_scaler
        temp_dict['category_equivalent_distance'] = distance
        temp_dict['category_equivalent_stdev'] = category_equivalent_stdev
        jsonable_dict[var] = temp_dict

    return jsonable_dict


def get_binary_mean(cdf: dict) -> float:
    t = cdf.get(True) or 0
    f = cdf.get(False) or 0
    return t / (t + f)


def fast_stdev(cdf: dict, distance: float) -> float:
    sorted_vals = sorted(cdf.values(), reverse=True)
    mean_sum = 0
    mean_distance = 0
    for val in sorted_vals:
        mean_sum = val * mean_distance
        mean_distance += distance

    mean = mean_sum / sum(sorted_vals)
    stdev_numerator = 0
    stdev_distance = 0
    for val in sorted_vals:
        stdev_numerator += ((stdev_distance - mean) ** 2) * val
        stdev_distance += distance

    return stdev_numerator / sum(sorted_vals)

