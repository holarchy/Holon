import common
import pandas as pd

weights = {
    'myip': 1.5356166194870193,
    'otherip': 1.5356166194870193,
    'myport': 1.88262193152,
    'otherport': 1.88262193152,
    'mysubnet': 2.84200602849924,
    'othersubnet': 2.84200602849924,
    'myflags': 2.3280000000000003,
    'otherflags': 2.3280000000000003,
    'mybytes': 1.12992,
    'otherbytes': 1.12992,
    'mypackets': 1.1823488000000002,
    'otherpackets': 1.1823488000000002,

    'protocol': 2.2,
    'category': 2.84,
    'flowtype': 2.2,
    'is_source': 0.870356080395215,
           }


def predict(flows: pd.DataFrame, subnet_priors: dict, ip_priors: dict, file_config: common.FileConfig):
    """

    :param flow: either 1 or 2 row dataframe representing a single flow between two machines. 1 row if only one machine
    is from the desired subnet; 2 rows of they both are.
    :param subnet_prior:
    :param ip_priors:
    :return:
    """

    predictions = list()
    for idx, row in flows.iterrows():
        my_prfx = file_config.uniflow_this_prfx
        ip_str = file_config.hierarchy[1]
        subnet_str = file_config.hierarchy[0]
        myip_str = my_prfx + ip_str
        mysubnet_str = my_prfx + subnet_str
        ip = row[myip_str]
        subnet = row[mysubnet_str]

        if row[file_config.uniflow_indicator] is True:
            rewrite_map = [(file_config.uniflow_this_prfx, file_config.biflow_src_prfx),
                           (file_config.uniflow_that_prfx, file_config.biflow_dst_prfx)]
        else:
            rewrite_map = [(file_config.uniflow_this_prfx, file_config.biflow_dst_prfx),
                           (file_config.uniflow_that_prfx, file_config.biflow_src_prfx)]

        if subnet_priors.get(subnet) is not None:
            predictions.append(predict_prior(row, subnet_priors.get(subnet), mysubnet_str, rewrite_map, subnet))

        if ip_priors.get(ip) is not None:
            predictions.append(predict_prior(row, ip_priors.get(ip), myip_str, rewrite_map, ip))

    biflow_values = dict() # need to pass in to write values to json
    uniflow_values = row.to_dict()
    for k, v in uniflow_values.items():
        biflow_values.update({k.replace(*rewrite_map[0]).replace(*rewrite_map[1]): v})
    prediction = combine_predictions(predictions, biflow_values)

    return prediction


def predict_prior(row: pd.Series, priors: dict, my_identifier, rewrite_map, subject_value):
    field_predictions = dict()
    for field, prior in priors.items():
        value = row[field]
        if value == value:  # check for NaN
            field_prediction = predict_parameter(field, value, prior, score_weight=weights[field])
            field_prediction.update({'id': my_identifier.replace(*rewrite_map[0]).replace(*rewrite_map[1]),
                                     'value': subject_value})
            field_predictions.update({field.replace(*rewrite_map[0]).replace(*rewrite_map[1]): field_prediction})

    return field_predictions


def predict_parameter(field: str, value: object, prior: dict, confidence_weight: float = 4, score_weight: float = 4):
    if field in common.numeric_vars():
        confidence_components = {'coeff_of_variation': conf_coeff_of_variation(stdev=prior['stdev'],
                                                                               mean=prior['mean']),
                                 'num_records': conf_num_records(
                                     prior_length=prior['prior_length'] - prior['nan_length'])}
        score_components = {'numeric_ppf': score_numeric_ppf(value=value,
                                                             cdf=prior['cdf']),
                            'dist_from_mean': score_dist_from_mean(value=value,
                                                                   stdev=prior['stdev'],
                                                                   mean=prior['mean'])}
    elif field in common.categorical_vars():
        confidence_components = {'category_equivalent_stdev': conf_category_equivalent_stdev(
            category_equivalent_stdev=prior['category_equivalent_stdev']),
                                 'num_records': conf_num_records(
                                     prior_length=prior['prior_length'] - prior['nan_length'])}
        score_components = {'categorical_proportion': score_categorical_proportion(value=value,
                                                                                   prior_length=prior['prior_length'] -
                                                                                                prior['nan_length'],
                                                                                   cdf=prior['cdf']),
                            'novelty': score_novelty(value=value,
                                                     cdf=prior['cdf'])}
    elif field in common.binary_vars():
        confidence_components = {'binary_mean': conf_binary_mean(cdf=prior['cdf']),
                                 'num_records': conf_num_records(
                                     prior_length=prior['prior_length'] - prior['nan_length'])}
        score_components = {'categorical_proportion': score_categorical_proportion(value=value,
                                                                                   prior_length=prior['prior_length'] -
                                                                                                prior['nan_length'],
                                                                                   cdf=prior['cdf']),
                            'novelty': score_novelty(value=value,
                                                     cdf=prior['cdf'])}
    else:
        raise KeyError(f'Cannot score the parameter {field}; it is neither defined as a numeric, categorical, or '
                       f'binary variable.')

    score = 1
    for s in [v for k, v in score_components.items()]:
        score *= s['value']

    confidence = 1
    for c in [v for k, v in confidence_components.items()]:
        confidence *= c['value']

    prediction = confidence ** confidence_weight * score ** score_weight

    return {'prediction': prediction,
            'score': score,
            'score_weight': score_weight,
            'score_components': score_components,
            'confidence': confidence,
            'confidence_weight': confidence_weight,
            'confidence_components': confidence_components}


def combine_predictions(predictions: list, biflow_data=None) -> dict:

    fields = set()
    for prediction in predictions:
        fields.update(prediction.keys())

    overall_predictions = list()
    for field in fields:
        field_predictions = list()
        for prediction in predictions:
            if field in prediction.keys():
                field_predictions.append(prediction[field])
        overall_predictions.append({'prediction': get_recursive_prediction_score(field_predictions),
                                    'id': field,
                                    'value': biflow_data.get(field),
                                    'subjects': field_predictions})
    return {'prediction': get_recursive_prediction_score(overall_predictions),
            'objects': overall_predictions}


def get_recursive_prediction_score(lst: list):

    score_remainder = 1
    overall_prediction = 0
    for v in lst:
        prediction = v['prediction']
        overall_prediction += score_remainder * prediction
        score_remainder = 1 - overall_prediction

    return overall_prediction


def conf_num_records(prior_length: int, scaler: float = 3) -> dict:
    if scaler < 0:
        raise ValueError(f'Scaler for conf_num_records should be > 0, got {scaler}.')

    value = (prior_length ** (1 / scaler) - 1) / (prior_length ** (1 / scaler))

    return {'value': value,
            'scaler': scaler}


def conf_coeff_of_variation(stdev: float, mean: float, scaler: float = 0.5) -> dict:
    if scaler < 0:
        raise ValueError(f'Scaler for conf_coeff_of_variation should be > 0, got {scaler}.')

    if mean != 0:  # todo review, avoiding divide by zero error
        coeff_of_var = stdev / mean
    else:
        coeff_of_var = 0
    value = 1 - coeff_of_var / (coeff_of_var + (1 / scaler))

    return {'value': value,
            'scaler': scaler}


def conf_category_equivalent_stdev(category_equivalent_stdev: float, scaler: float = 1) -> dict:
    if scaler < 0:
        raise ValueError(f'Scaler for conf_category_equivalent_stdev should be > 0, got {scaler}.')

    value = 1 - (category_equivalent_stdev / (category_equivalent_stdev + (1 / scaler)))

    return {'value': value,
            'scaler': scaler}


def conf_binary_mean(cdf: dict, scaler: float = 2) -> dict:
    # the mean of a binary distribution equals the number of true values / the number of values
    if scaler < 0:
        raise ValueError(f'Scaler for conf_binary_mean should be > 0, got {scaler}.')

    t = cdf.get('true') or 0
    f = cdf.get('false') or 0
    mean = t / (t + f)
    adjusted_mean = abs(mean - 0.5)
    value = (2 * adjusted_mean) ** scaler

    return {'value': value,
            'scaler': scaler}


def score_dist_from_mean(value: float, stdev: float, mean: float, scaler: float = 2) -> dict:
    if scaler < 0:
        raise ValueError(f'Scaler for score_dist_from_mean should be > 0, got {scaler}.')

    if stdev != 0:  # todo review this, combat divide by 0 error
        distance = abs(value - mean) / stdev
    else:
        distance = abs(value - mean)
    value = ((distance + 1) ** (1 / scaler) - 1) / ((distance + 1) ** (1 / scaler))

    return {'value': value,
            'scaler': scaler}


def score_novelty(value: str, cdf: dict, scaler: float = 2) -> dict:
    if scaler < 0:
        raise ValueError(f'Scaler for score_novelty should be > 0, got {scaler}.')

    num_unique_values = len(cdf)
    if value in cdf:
        value = 1 - (0.5 * num_unique_values / (num_unique_values + (1 / scaler)))
    else:
        value = 0.5 * num_unique_values / (num_unique_values + (1 / scaler))

    return {'value': value,
            'scaler': scaler}


def score_categorical_proportion(value: str, prior_length: int, cdf: dict, scaler: float = 40) -> dict:
    if scaler < 0:
        raise ValueError(f'Scaler for score_categorical_proportion should be > 0, got {scaler}.')

    if type(value) == bool:
        value = str(value).lower()  # todo review. workaround because json stores bool as lowercase str
    category_size = cdf.get(value)
    if category_size is None:
        category_size = 0
    categorical_proportion = category_size / prior_length
    value = (1 - categorical_proportion) ** scaler

    return {'value': value,
            'scaler': scaler}


def score_numeric_ppf(value: float, cdf: dict, scaler: float = 10) -> dict:
    if scaler < 0:
        raise ValueError(f'Scaler for score_numeric_ppf should be > 0, got {scaler}.')

    ppf = get_numeric_ppf(value, cdf)
    value = (2 * abs((ppf - 0.5))) ** scaler

    return {'value': value,
            'scaler': scaler}


def get_numeric_ppf(value: float, cdf: dict) -> float:
    if cdf.get('1') is not None:  # todo remake prior and delete this code
        cdf['1.0'] = cdf['1']
        del cdf['1']

    if cdf.get('0') is not None:  # todo remake prior and delete this code
        cdf['0.0'] = cdf['0']
        del cdf['0']

    if value in cdf.values():
        appropriate_values = [float(k) for k, v in cdf.items() if v == value]
        return min(appropriate_values)
    elif value > cdf['1.0']:
        return 1
    elif value < cdf['0.0']:
        return 0
    else:
        lower_key = max([float(k) for k, v in cdf.items() if v < value])
        upper_key = min([float(k) for k, v in cdf.items() if v > value])

        lower_value = cdf[str(lower_key)]
        upper_value = cdf[str(upper_key)]

        return lower_key + ((upper_key - lower_key) / (upper_value - lower_value)) * (value - lower_value)
