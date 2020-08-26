from flask import Flask, render_template, flash, abort, redirect, url_for, request
import os
import common
import json
import numbers
import urllib.parse
import pandas as pd
from datetime import datetime
from math import log10, floor

base_dir = '/home/nick/Data/_ensembles'
app = Flask(__name__)
app.config['ENV'] = 'development'
app.config['DEBUG'] = True
app.config['TESTING'] = True

app.config.from_mapping(
    SECRET_KEY='dev'
)

# predictions_home_dir = os.path.join(base_dir, 'outlier-predictions-2019_11_13-15_38_28')
predictions_home_dir = os.path.join(base_dir, 'outlier-predictions-2020_01_03-11_15_41')
file_config = common.load_file_config(predictions_home_dir)
labels_dir = os.path.join(predictions_home_dir, 'labels')
# priors_parent_dir = os.path.join(base_dir, 'priors-2019_11_12-19_33_13')
priors_parent_dir = os.path.join(base_dir, 'priors-2019_12_30-18_30_22')
predictions_dir = os.path.join(predictions_home_dir, 'predictions')
priors_dir = os.path.join(priors_parent_dir, 'priors')
prediction_summary = pd.read_csv(os.path.join(predictions_home_dir, 'summary.csv'))
prediction_summary = prediction_summary.sort_values('prediction', ascending=False)
prediction_summary = prediction_summary.reset_index()


def get_flow(flow):
    file = os.path.join(predictions_dir, flow + '.json')
    if not os.path.isfile(file):
        flash(f'{flow} was not found.')
        abort(404)
    with open(file) as f:
        flow = json.load(f)
    return flow


def make_label(flow, username, threat_level, classifier, description):

    if not os.path.isdir(labels_dir):  # make label directory if it doesn't exist.
        os.mkdir(labels_dir)

    flow_data = get_flow(flow)
    prediction_values = list()
    for obj in flow_data['objects']:
        prediction_values.append((obj['id'], obj['value'], obj['prediction']))

    label_file = os.path.join(labels_dir, flow + '.json')  # get filename based on flow name

    if os.path.isfile(label_file):
        # jsn = []
        with open(label_file, 'r') as f:
            jsn = json.load(f)  # if file already exists, get json.
    else:
        jsn = []

    dict = {'userName': username,
            'threatLevel': threat_level,
            'classifier': classifier,
            'description': description,
            'timestamp': str(datetime.now()),
            'version': common.__version__,
            'data': prediction_values}

    jsn.append(dict)

    with open(label_file, 'w') as f:
        json.dump(jsn, f)


def remove_label(flow, index):
    label_file = os.path.join(labels_dir, flow + '.json')  # get filename based on flow name
    with open(label_file, 'r') as f:
        jsn = json.load(f)  # if file already exists, get json.

    del jsn[index]

    with open(label_file, 'w') as f:
        json.dump(jsn, f)


def get_labels(flow):
    label_file = os.path.join(labels_dir, flow + '.json')  # get filename based on flow name
    if os.path.isfile(label_file):
        with open(label_file, 'r') as f:
            jsn = json.load(f)  # if file already exists, get json.
    else:
        jsn = []

    return jsn


def round_structure(x, sig=2):
    if isinstance(x, numbers.Number):
        if x == 0 or x != x:  # alo check for NaN
            return 0
        return round(x, sig - int(floor(log10(abs(x)))) - 1)
    elif isinstance(x, dict):
        dct = dict()
        for k, v in x.items():
            dct[k] = round_structure(v, sig)
        return dct
    elif isinstance(x, list):
        lst = list()
        for itm in x:
            lst.append(round_structure(itm, sig))
        return lst
    elif type(x) in (str, bool):
        return x
    else:
        raise TypeError


class PredictionTrace(object):
    levels = ['Flow', 'Object', 'Subject']

    def __init__(self, flow, obj=None, subject=None):
        if flow is None:
            raise ValueError(f'Flow parameter cannot be None')

        field_predictions = None

        flow = urllib.parse.unquote(flow)
        jsn = get_flow(flow)
        jsn = round_structure(jsn)
        raw_data = jsn.get('raw_data')

        self.flow = flow
        self.biflow_object = obj
        self.subject = subject
        self.raw_data = raw_data

        level = self.levels[0]
        prediction_trace = [(level, 'Outlier Score', '', jsn['prediction'])]
        prediction_list = 'objects'
        prediction_field = 'id'
        if obj is not None:
            obj = urllib.parse.unquote(obj)
            jsn = self.get_level_json(jsn, obj, prediction_list, prediction_field)
            level = self.levels[1]
            prediction_trace.append((level, obj, jsn['value'], jsn['prediction']))
            prediction_list = 'subjects'
            prediction_field = 'id'
            if subject is not None:
                subject = urllib.parse.unquote(subject)
                jsn = self.get_level_json(jsn, subject, prediction_list, prediction_field)
                level = self.levels[2]
                prediction_trace.append((level, subject, jsn['value'], jsn['prediction']))
                prediction_list = None
                prediction_field = None
                field_predictions = jsn

        predictions = []
        if prediction_field is not None:
            for identifier in jsn[prediction_list]:
                predictions.append({'id': identifier[prediction_field],
                                    'pred': identifier['prediction'],
                                    'val': identifier.get('value')})

        self.level = level
        self.prediction_trace = prediction_trace
        self.predictions = sorted(predictions, key=lambda i: i['pred'], reverse=True)
        self.field_predictions = field_predictions

    @property
    def my_direction(self):
        return file_config.my_direction(self.subject)

    @property
    def their_direction(self):
        return file_config.their_direction(self.subject)

    @property
    def field_value(self):
        if self.biflow_object == file_config.uniflow_indicator:  # special case
            return str(self.my_direction == file_config.biflow_src_prfx).lower()
        else:
            return self.raw_data[self.biflow_object]

    @property
    def field_prior(self):
        if self.biflow_object is None:
            raise ValueError(f'Can only pull prior based on a field.')

        if self.subject.endswith(file_config.hierarchy[0]):  # subnet
            path = os.path.join(priors_dir,
                                self.raw_data[self.my_direction + file_config.hierarchy[0]])

        elif self.subject.endswith(file_config.hierarchy[1]):  # ip
            path = os.path.join(priors_dir,
                                self.raw_data[self.my_direction + file_config.hierarchy[0]],
                                self.raw_data[self.my_direction + file_config.hierarchy[1]])
        else:
            raise ValueError(f'Did not recognize level "{self.subject}"')

        file = os.path.join(path, '.json')
        if not os.path.isfile(file):
            raise ValueError(f'Priors file {file} was not found.')
        with open(file) as f:
            prior = json.load(f)

        field_prior = prior[self.uniflow_object]
        return field_prior

    @property
    def uniflow_object(self):
        if self.subject is None:
            raise ValueError(f'Cannot call uniflow_object without both a _subject_ (ex. dst.ip) and an _object_ (ex. '
                             f'src.bytes).')

        if self.biflow_object.startswith(self.my_direction):
            return self.biflow_object.replace(self.my_direction, file_config.uniflow_this_prfx)
        elif self.biflow_object.startswith(self.their_direction):
            return self.biflow_object.replace(self.their_direction, file_config.uniflow_that_prfx)
        else:
            return self.biflow_object

    @property
    def child_level(self):
        this = self.level
        print(this)
        print(self.levels[2])
        if this == self.levels[2]:
            raise ValueError(f'"Subject" level has no child.')
        return self.levels[self.levels.index(this) + 1]

    def build_url(self, lvl):

        if lvl not in self.levels:
            raise ValueError(f'build_url requires one of the 4 defined levels')

        segments = ['/prediction', urllib.parse.quote(self.flow)]
        if lvl != self.levels[0]:
            segments.append(urllib.parse.quote(self.biflow_object))
            if lvl != self.levels[1]:
                segments.append(urllib.parse.quote(self.subject))
        return '/'.join(segments)

    @staticmethod
    def get_level_json(jsn, value, prediction_list, prediction_field):
        level_json = [p for p in jsn.get(prediction_list) if p[prediction_field] == value]
        if len(level_json) == 0:
            flash(f'{level_json} was not found.')
            abort(404)
        return level_json.pop()

    @property
    def chart_data(self):
        primary_color = '#007bff'
        secondary_color = '#6c757d'
        max_columns = 15
        cdf = self.field_prior['cdf']
        if self.uniflow_object in common.numeric_vars():
            typ = 'scatter'
            data = [{'x': float(k), 'y': v} for k, v in cdf.items()]
            full_data = {'datasets': [{'label': self.uniflow_object,
                                       'backgroundColor': secondary_color,
                                       'data': data},
                                       {'label': self.field_value,
                                       'backgroundColor': primary_color,
                                       'showLine': 'true',
                                       'borderColor': primary_color,
                                       'data': [{'x': 0, 'y': self.field_value},
                                                {'x': 1, 'y': self.field_value}]},
                                      ]}
        elif self.uniflow_object in common.binary_vars() or self.uniflow_object in common.categorical_vars():
            typ = 'bar'
            ln = len(cdf)
            ix = None
            if self.field_value in cdf.keys():
                ix = list(cdf.keys()).index(self.field_value)
            if ln < max_columns:
                indexes = list(range(0, ln))
            else:
                if ix is None or ix < 10 or ix > ln - 4:
                    indexes = list(range(0, 10)) + [f'MANY\n({ln - 14})'] + list(range(ln - 4, ln))
                else:
                    indexes = list(range(0, 10)) + [f'MANY\n({ix - 10})'] + [ix] + [f'MANY\n({ln - ix - 3})'] + list(range(ln - 3, ln))
            labels = [list(cdf.keys())[idx] if type(idx) == int else idx for idx in indexes]
            data = [list(cdf.values())[idx] if type(idx) == int else 0 for idx in indexes]
            colors = [primary_color if itm == self.field_value else secondary_color for itm in labels]
            full_data = {'labels': labels,
                         'datasets': [{'label': self.uniflow_object,
                                       'backgroundColor': colors,
                                       'data': data}]}
        else:
            raise ValueError(f'Field does not seem to be valid, has value {self.uniflow_object}')

        chart_data = {'type': typ,
                      'data': full_data,
                      'options': {
                          'legend': {'display': 'false'},
                          'scales': {'yAxes': [{'ticks': {'min': 0}}]}}}
        return chart_data


@app.route('/')
@app.route('/summary/')
@app.route('/prediction/')
def index():
    return redirect(url_for('summary', page_num=1))


@app.route('/summary/<int:page_num>')
def summary(page_num=1):
    results_per_page = 10
    i = (page_num - 1) * results_per_page
    if i > len(prediction_summary):
        abort(404)
    predictions = []
    n = 0
    while n < results_per_page and i < len(prediction_summary):
        p = prediction_summary.loc[i]
        id = p['filename'].replace('.json','')
        data = id.split('_')
        ts = datetime.fromtimestamp(int(data[0])/1000)
        pred = round_structure(p['prediction'])
        labels = get_labels(id)
        if len(labels):
            classification = labels[0]['threatLevel']
        else:
            classification = ''
        predictions.append({'id': id, 'timestamp': ts, 'src_ip': data[1], 'src_port': p['src.port'], 'dst_ip': data[2],
                            'dst_port': p['dst.port'], 'classification': classification, 'pred': pred, 'index': i})
        i += 1
        n += 1

    last_page = floor(len(prediction_summary) / results_per_page) + 1
    nav_display = dict()
    nav_display.update({1: '&laquo;', last_page: '&raquo;'})
    if page_num not in (1, last_page):
        nav_display.update({n: str(n) for n in list(range(page_num - 1, page_num + 2))})
    if page_num <= 3:
        nav_display.update({n: str(n) for n in list(range(1,4))})
    if page_num >= last_page - 3:
        nav_display.update({n: str(n) for n in list(range(last_page-2, last_page+1))})

    nav_display = dict(sorted(nav_display.items()))

    return render_template('summary.html', predictions=predictions, page_num=page_num, nav_display=nav_display)


def resolve_user_label(flow, request):
    if request.method == "POST":
        if request.form.get('threatLevel') is not None:  # if user added new label
            make_label(flow, username=request.form.get('userName'), threat_level=request.form.get('threatLevel'),
                       classifier=request.form.get('classifier'), description=request.form.get('description'))
        else:  # if user trying to delete label
            i = 1
            while i <= len(get_labels(flow)):
                if request.form.get(str(i)) is not None:
                    print(i)
                    remove_label(flow, i-1)
                i += 1


@app.route('/prediction/<flow>', methods=['GET', 'POST'])
@app.route('/prediction/<flow>/<object>', methods=['GET', 'POST'])
def flow_prediction(flow, object=None):
    resolve_user_label(flow, request)
    trace = PredictionTrace(flow, object)
    return render_template('level_explorer.html', trace=trace, labels=get_labels(flow))


@app.route('/prediction/<flow>/<object>/<subject>', methods=['GET', 'POST'])
def field_prediction(flow, object, subject):
    resolve_user_label(flow, request)
    trace = PredictionTrace(flow, object, subject)
    return render_template('field_explorer.html', trace=trace, labels=get_labels(flow))


@app.route('/refs')
def refs():
    return render_template('references.html')


@app.route('/admin/')
def admin():
    return redirect(url_for('admin_data'))


@app.route('/admin/data', methods=['GET', 'POST'])
def admin_data():
    def get_metadata(dir, pattern):
        metadata = list()
        for subdir in os.listdir(dir):
            path = os.path.join(dir, subdir)
            if subdir.startswith(pattern) and os.path.isdir(path):
                filepath = os.path.join(path, 'metadata.json')
                if os.path.isfile(filepath):
                    with open(filepath) as f:
                        jsn = json.load(f)
                    md = {'directory': os.path.basename(dir),
                          'md5': jsn.get('md5'),
                          'filename': jsn.get('filename'),
                          'size (GB)': jsn.get('size (GB)'),
                          'number of rows': jsn.get('number of rows'),
                          'start date': jsn.get('start date'),
                          'end date': jsn.get('end date'),
                          'package version': jsn.get('package version'),
                          }
                    metadata.append(md)
        return metadata

    prior_metadata = get_metadata(base_dir, 'priors')
    pred_metadata = get_metadata(base_dir, 'outlier-predictions')
    raw_metadata = get_metadata(base_dir, 'raw-data')

    return render_template('admin_data.html', pred_metadata=pred_metadata, prior_metadata=prior_metadata, raw_metadata=raw_metadata)


@app.route('/admin/labels', methods=['GET', 'POST'])
def admin_labels():
    return render_template('admin_labels.html')


@app.route('/admin/file-config', methods=['GET', 'POST'])
def admin_data_config():
    return render_template('admin_file_config.html')


@app.errorhandler(404)
def page_not_found(e):
    flash(f'404: Page not found.')
    return render_template('base.html')
