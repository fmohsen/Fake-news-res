# Program: Machine learning framework
# Author: Bedir Chaushi, Kevin Wang
# Description: This program creates machine learning models and provides result of text classification dataset.
import os

# import necessarry modules and methods

from helper import get_results, getDatasetWithSignificantFeatures, feature_extraction, word_stemming, \
    remove_punctuation_stopwords, shuffle_csv, keep_columns, parse_data, drop_na, create_pdf, vectorize_feature, \
    mergeDatasetWithLabels
from datetime import date
import time

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import dash_daq as daq

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

import pickle

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# import external stylesheet for the web app
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

colors = {"graphBackground": "#F5F5F5",
          "background": "#ffffff", "text": "#000000"}

# models and data types used as dropdown options
models = ["Logistic Regression",
          "Decision Trees",
          "Nearest Neighbour",
          "Gradient Boosting",
          "Naive Bayes"]

data_type = [{"label": "Textual Data", "value": "Textual Data"}, {
    "label": "Numerical Data (this feature is coming soon!)", "value": "Numerical Data", "disabled": True}]

# the python dash app layout

app.layout = html.Div(
    [
        html.Div([html.P(
            "Welcome to machine learning evaluation app. This app is intended to fullfil one of the proposed goals of the Master Thesis with the title: 'Evaluating machine learning algorithms for English fake news detection'"),
            html.P(
                "The goal of this application to generate predictions results based on a specific user uploaded dataset. The user can select between different parameters, and upon providing results, one can download a pdf report of the results and export the machine learning model"),
            html.H5("To get started, please click on the button below."),
            html.Button('Start the application',
                        id='start-app-button', n_clicks=0, disabled=False)
        ], style={'display': 'block'}, id="start-app", className="start-app-container"),
        html.Div(
            [
                dcc.Upload(
                    id="upload-data",
                    children=html.Div(
                        ["Drag and Drop or ", html.A("Select Files")]),
                    style={
                        "width": "100%",
                        "height": "60px",
                        "lineHeight": "60px",
                        "borderWidth": "1px",
                        "borderStyle": "dashed",
                        "borderRadius": "5px",
                        "textAlign": "center",
                        "margin": "10px",
                    },
                    # Allow multiple files to be uploaded
                    multiple=True,
                ),
                html.Div(id="output-data-upload"),
                html.Div([
                    html.Div([
                        html.P("Select label from feature list",
                               className="control_label"),
                        html.P("Select which feature from drowdown is identifying the label of specific row",
                               className="small-info"),
                        dcc.Dropdown(
                            id="select_label",
                            options=[{'label': x, 'value': x} for x in models],
                            className="dcc_control",

                        ),
                        html.Br(),
                        html.P("Algorithms to use", className="control_label"),
                        html.P("Select multiple algorithms alongside their parameters for analysis",
                               className="small-info"),
                        dcc.Dropdown(
                            id="select_models",
                            options=[{'label': x, 'value': x} for x in models],

                            multi=True,
                            clearable=False,
                            className="dcc_control",

                        ),
                        # logistic regression parameters
                        html.Div([
                            html.B("Select logistic regression parameters",
                                   className="lr_title", style={"text-decoration": "underline"}),
                            html.P("Select solver",
                                   className="select_lr_solver"),
                            html.P(
                                "the optimization algorithm used to find the optimal coefficients (Note: Not all solvers support all norms of penalty. Check sci-kit learn documentation to find out which solvers support which subset of parameters)",
                                className="small-info"),

                            dcc.RadioItems(["liblinear", "saga", "newton-cg", "lbfgs"], "lbfgs",
                                           inline=True, id="lr_solver"),

                            html.P("Select regularization (c)",
                                   className="select_lr_c"),
                            html.P(
                                "Inverse of regularization strength; must be a positive float. Smaller values specify stronger regularization.",
                                className="small-info"),

                            daq.Slider(id='lr_c', min=0, max=1.0, value=1.0, handleLabel={
                                "showCurrentValue": True, "label": "value"}, step=0.1),

                            html.P("Select penalty",
                                   className="select_lr_penalty"),
                            html.P("Specify the norm of the penalty",
                                   className="small-info"),

                            dcc.RadioItems(["none", "l2", "l1", "elasticnet"], "l2",
                                           inline=True, id="lr_penalty"),

                            html.P("Select maximum iterration",
                                   className="select_lr_max_iter"),
                            html.P("Maximum number of iterations taken for the solvers to converge.",
                                   className="small-info"),

                            daq.Slider(id='lr_max_iter', min=100, max=1000, value=100, handleLabel={
                                "showCurrentValue": True, "label": "value"}, step=100),
                            html.Hr()

                        ], style={"display": "none"}, id="lr_parameters"),

                        # decision tree parameters
                        html.Div([
                            html.B("Select decision tree parameters",
                                   className="dt_title", style={"text-decoration": "underline"}),
                            html.P("Select criterion",
                                   className="select_lr_criterion"),
                            html.P("The function to measure the quality of a split.",
                                   className="small-info"),

                            dcc.RadioItems(["gini", "entropy", "log_loss"], "gini",
                                           inline=True, id="dt_criterion"),

                            html.P("Select minimum split",
                                   className="select_dt_min_split"),
                            html.P("The minimum number of samples required to split an internal node",
                                   className="small-info"),

                            daq.Slider(id='dt_min_samples_split', min=2, max=10, value=2, handleLabel={
                                "showCurrentValue": True, "label": "value"}, step=1),

                            html.P("Select splitter",
                                   className="select_dt_splitter"),
                            html.P("The strategy used to choose the split at each node.",
                                   className="small-info"),

                            dcc.RadioItems(["best", "random"], "best",
                                           inline=True, id="dt_splitter"),

                            html.P("Select maximum depth",
                                   className="select_lr_max_iter"),
                            html.P("The maximum depth of the tree.",
                                   className="small-info"),

                            daq.Slider(id='dt_max_depth', min=1, max=50, value=30, handleLabel={
                                "showCurrentValue": True, "label": "value"}, step=1),
                            html.Hr()

                        ], style={"display": "none"}, id="dt_parameters"),

                        # nearest neighbour  parameters
                        html.Div([
                            html.B("Select Nearest neighbour parameters",
                                   className="knn_title", style={"text-decoration": "underline"}),
                            html.P("Select algorithm",
                                   className="select_knn_algorithm"),
                            html.P("Algorithm used to compute the nearest neighbors:",
                                   className="small-info"),

                            dcc.RadioItems(["auto", "ball_tree", "kd_tree", "brute"], "auto",
                                           inline=True, id="knn_algorithm"),

                            html.P("Select number of neighbours",
                                   className="select_knn_neighbours"),
                            html.P("Number of neighbors to use by default for kneighbors queries.",
                                   className="small-info"),

                            daq.Slider(id='knn_neighbours', min=3, max=8, value=5, handleLabel={
                                "showCurrentValue": True, "label": "value"}, step=1),

                            html.P("Select weights",
                                   className="select_knn_weights"),
                            html.P("Weight function used in prediction.",
                                   className="small-info"),

                            dcc.RadioItems(["uniform", "distance"], "uniform",
                                           inline=True, id="knn_weight"),

                            html.P("Select distance calculation method",
                                   className="select_knn_distance"),
                            html.P(
                                "When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2. ",
                                className="small-info"),

                            dcc.RadioItems(["1", "2"], "2",
                                           inline=True, id="knn_distance"),
                            html.Hr()

                        ], style={"display": "none"}, id="knn_parameters"),

                        # gradient boosting parameters
                        html.Div([
                            html.B("Select gradient boosting parameters", className="gb_title", style={
                                "text-decoration": "underline"}),

                            html.P("Select criterion",
                                   className="select_gb_criterion"),
                            html.P(
                                "The function to measure the quality of a split..", className="small-info"),

                            dcc.RadioItems(["friedman_mse", "squared_error"], "friedman_mse",
                                           inline=True, id="gb_criterion"),

                            html.P("Select loss function",
                                   className="select_gb_loss_function"),
                            html.P("The loss function to be optimized.",
                                   className="small-info"),

                            dcc.RadioItems(["log_loss", "deviance", "exponential"], "log_loss",
                                           inline=True, id="gb_loss_function"),

                            html.P("Select learning rate",
                                   className="select_gb_learning_rate"),
                            html.P("Learning rate shrinks the contribution of each tree by learning_rate",
                                   className="small-info"),

                            daq.Slider(id='gb_learning_rate', min=0, max=2.0, value=0.1, handleLabel={
                                "showCurrentValue": True, "label": "value"}, step=0.1),

                            html.P("Select max depth",
                                   className="select_gb_max_depth"),
                            html.P(
                                "Maximum depth of the individual regression estimators. The maximum depth limits the number of nodes in the tree. Tune this parameter for best performance",
                                className="small-info"),

                            daq.Slider(id='gb_max_depth', min=0, max=20, value=3, handleLabel={
                                "showCurrentValue": True, "label": "value"}, step=1),

                            html.P("Select minimum samples split",
                                   className="select_gb_min_samples_split"),
                            html.P("The minimum number of samples required to split an internal node",
                                   className="small-info"),

                            daq.Slider(id='gb_min_samples_split', min=2, max=10, value=2, handleLabel={
                                "showCurrentValue": True, "label": "value"}, step=1),

                            html.Hr()

                        ], style={"display": "none"}, id="gb_parameters"),

                        html.Div([
                            html.B("Select Naive Bayes parameters",
                                   className="nb_title", style={"text-decoration": "underline"}),
                            html.P("Select algorithm",
                                   className="select_nb_algorithm"),
                            html.P("Variation of Naive Bayes algorithm:",
                                   className="small-info"),

                            dcc.RadioItems(["Multinomial", "Bernoulli"], "Bernoulli",
                                           inline=True, id="nb_algorithm"),

                            html.P("Apply Transfer Learning", className="transfer_learning"),
                            html.P("Leverage pre-trained models(if any) and learned patterns to enhance the efficiency of the task.", className="small-info"),
                            dcc.RadioItems(["Yes", "No"], "No", inline=True, id="toggle_transfer_learning"),

                            html.P("Input the location of the pretrained model:", id="pre_location"),
                            #dcc.Upload(id="model_location", children=html.Div(["Drag and Drop or ", html.A("Select Model")])),
                            dcc.Input(id="model_location"),

                            html.P("Select value for Alpha",
                                   className="select_alpha_value"),
                            html.P("Additive (Laplace/Lidstone) smoothing parameter.",
                                   className="small-info"),

                            daq.Slider(id='alpha_value', min=0, max=1, value=1, handleLabel={
                                "showCurrentValue": True, "label": "value"}, step=0.01),

                            html.P("Estimate Class Priors from Data",
                                   className="check_fit_prior"),
                            html.P(
                                "Whether to learn class prior probabilities or not. If not selected, a uniform prior will be used.",
                                className="small-info"),

                            dcc.RadioItems(["Enabled", "Disabled"], "Enabled", id="fit_prior"),

                            html.P("Customize class priorities",
                                   className="select_class_prior"),
                            html.P(
                                "Prior probabilities of the classes. If specified, the priors are not adjusted according to the data.",
                                className="small-info"),
                            dcc.RadioItems(
                                id='toggle_class_prior',
                                options=[
                                    {'label': 'True', 'value': True},
                                    {'label': 'False', 'value': False}
                                ],
                                value=False
                            ),

                            daq.Slider(
                                id='slider1',
                                min=0,
                                max=1,
                                step=0.01,
                                value=0.5,
                                handleLabel={"showCurrentValue": True, "label": "value"},
                                updatemode='drag'
                            ),
                            daq.Slider(
                                id='slider2',
                                min=0,
                                max=1,
                                step=0.01,
                                value=0.5,
                                handleLabel={"showCurrentValue": True, "label": "value"},
                                updatemode='drag'
                            ),

                            html.Hr()

                        ], style={"display": "none"}, id="nb_parameters"),

                        html.Br(),
                        html.P("Data types", className="control_label"),
                        html.P("Choose what type of data you have uploaded",
                               className="small-info"),

                        dcc.Dropdown(
                            id="select_type_data",
                            placeholder="Select what type of data you have",
                            options=data_type,
                            className="dcc_control",

                        ),
                        html.Div([
                            html.Br(),
                            html.P("Choose vectorizer algorithm",
                                   className="control_label"),
                            html.P("Select one of the vectorizer technique for the textual content",
                                   className="small-info"),

                            dcc.RadioItems(["Tfidf", "Feature Extraction", "Hybrid Model"], "Tfidf",
                                           inline=True, id="textual_selection"),
                        ], style={"display": "none"}, id="vecAlg"),

                        html.Div([
                            html.Br(),
                            html.P("Choose which feature to vectorize",
                                   className="control_label"),
                            html.P(
                                "Choose which is the feature desired for analysis. If it is news data, select the article headline or content if aplicable",
                                className="small-info"),
                            dcc.Dropdown(
                                id="select_feature",
                                options=[{'label': x, 'value': x}
                                         for x in models],
                                className="dcc_control", ),
                        ], style={"display": "none"}, id="featureToUse"),

                        html.Div([
                            html.Br(),
                            html.P("Choose p-value for most significant features",
                                   className="control_label"),
                            html.P(
                                "p-value determines the statistical significance of the features. Note: The lower the p-value , the greater the statistical significance of the observed difference",
                                className="small-info"),
                            dcc.RadioItems(['0.05', '0.005',
                                            '0.01', '0.001'], '0.001', inline=True, id="numerical_selection"),
                        ], style={'display': 'none'}, id="pValue"),

                        html.Br(),
                        html.P("Train test split", className="control_label"),
                        html.P(
                            "Determine the trainig and test data split percentage. The number selected in slider indicates training data split percentage",
                            className="small-info"),

                        daq.Slider(
                            id='slider',
                            min=0,
                            max=100,
                            value=75,
                            handleLabel={
                                "showCurrentValue": True, "label": "SPLIT"},
                            step=5,
                            labelPosition="bottom"
                        ),

                        html.Div([
                            html.Button('Generate predictions',
                                        id='submit-val', n_clicks=0, disabled=False)
                        ], className="align-center")

                    ], className="float-child"),
                    html.Div([

                    ], className="float-child align-center", id="show-results"),
                    html.Div([
                        html.H5(
                            "The seleceted Machine learning models are exported as pickle files. You can find them in this app directory in folder named pickles/.",
                            className="pickles"),
                        html.Button('Export results to PDF',
                                    id='export-pdf', n_clicks=0, disabled=False), dcc.Download(id="download"),
                    ], className="align-center", id="export-pdf-div", style={'display': 'none'})

                ], className="float-container"),
                html.Div(id='trigger', children=0, style=dict(display='none')),

            ], style={'display': 'none'}, id="main-container"
        )
    ])


@app.callback([Output('pre_location', 'style'),
               Output('model_location', 'style')],
              Input('toggle_transfer_learning', 'value'))
def toggle_transfer(value):
    if value == "Yes":
        return {'display': 'block'}, {'display': 'block'}
    else:
        return {'display': 'none'}, {'display': 'none'}

@app.callback(
    [Output('slider1', 'style'),
     Output('slider2', 'style')],
    Input('toggle_class_prior', 'value'))
def toggle_sliders(value):
    if value:
        return {'display': 'block'}, {'display': 'block'}
    else:
        return {'display': 'none'}, {'display': 'none'}


@app.callback(
    Output('slider1', 'value'),
    Input('slider2', 'value'))
def update_slider1(slider2_value):
    return 1 - slider2_value


@app.callback(
    Output('slider2', 'value'),
    Input('slider1', 'value'))
def update_slider2(slider1_value):
    return 1 - slider1_value


# callback reads uploaded dataset and outputs it using pagination
@app.callback(
    Output("output-data-upload", "children"),
    [Input("upload-data", "contents"), Input("upload-data", "filename")],
)
def update_table(contents, filename):
    table = html.Div()

    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_data(contents, filename)
        table = html.Div(
            [
                html.H5(filename),
                dash_table.DataTable(

                    data=df.to_dict("rows"),
                    columns=[{"name": i, "id": i} for i in df.columns],
                    page_action="native",
                    page_current=0,
                    page_size=5,
                    # left align text in columns for readability
                    style_cell={'textAlign': 'left',
                                'overflow': 'hidden',
                                'textOverflow': 'ellipsis',
                                'maxWidth': 0}

                )
            ]
        )

    return table


# when logistic regression is selected, its parameters are shown
@app.callback(
    Output(component_id='lr_parameters', component_property='style'),
    [Input(component_id='select_models', component_property='value')])
def show_hide_lr_parameters(dropdown_value):
    if 'Logistic Regression' in dropdown_value:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


# when decision tree is selected, its parameters are shown
@app.callback(
    Output(component_id='dt_parameters', component_property='style'),
    [Input(component_id='select_models', component_property='value')])
def show_hide_lr_parameters(dropdown_value):
    if 'Decision Trees' in dropdown_value:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


# when nearest neighbour is selected, its parameters are shown
@app.callback(
    Output(component_id='knn_parameters', component_property='style'),
    [Input(component_id='select_models', component_property='value')])
def show_hide_lr_parameters(dropdown_value):
    if 'Nearest Neighbour' in dropdown_value:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


# when gradient boosting is selected, its parameters are shown
@app.callback(
    Output(component_id='gb_parameters', component_property='style'),
    [Input(component_id='select_models', component_property='value')])
def show_hide_lr_parameters(dropdown_value):
    if 'Gradient Boosting' in dropdown_value:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


# when naive bayes is selected, its parameters are shown
@app.callback(
    Output(component_id='nb_parameters', component_property='style'),
    [Input(component_id='select_models', component_property='value')])
def show_hide_lr_parameters(dropdown_value):
    if 'Naive Bayes' in dropdown_value:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


# start app button callback
@app.callback(
    [Output(component_id='main-container', component_property='style'),
     Output(component_id='start-app', component_property='style')],
    [Input('start-app-button', 'n_clicks')])
def show_hide_pvalue(button_clicked):
    if button_clicked:
        return ({'display': 'block'}, {'display': 'none'})


# displays when numerical data selected (TO DO)
@app.callback(
    Output(component_id='pValue', component_property='style'),
    [Input(component_id='select_type_data', component_property='value')])
def show_hide_pvalue(dropdown_value):
    if dropdown_value == 'Numerical Data':
        return {'display': 'block'}


# displays when textual data selected
@app.callback(
    Output(component_id='vecAlg', component_property='style'),
    [Input(component_id='select_type_data', component_property='value')])
def show_hide_vecalg(dropdown_value):
    if dropdown_value == 'Textual Data':
        return {'display': 'block'}


# dropdown to select the column of interest from feature set
@app.callback(
    Output(component_id='featureToUse', component_property='style'),
    [Input(component_id='select_type_data', component_property='value')])
def show_hide_vecalg(dropdown_value):
    if dropdown_value == 'Textual Data':
        return {'display': 'block'}


# callback that reads all inputs and creates the pdf
@app.callback(
    Output(component_id='download', component_property='data'),
    [State(component_id='select_type_data', component_property='value'),
     State(component_id='select_models', component_property='value'),
     State(component_id='textual_selection', component_property='value'),
     State(component_id='select_feature', component_property='value'),
     State(component_id='numerical_selection', component_property='value'),
     State(component_id='slider', component_property='value'),
     State(component_id='select_label', component_property='value'),
     State("upload-data", "contents"), State("upload-data", "filename"),
     State(component_id='lr_solver', component_property='value'),
     State(component_id='lr_c', component_property='value'),
     State(component_id='lr_penalty', component_property='value'),
     State(component_id='lr_max_iter', component_property='value'),
     State(component_id='dt_criterion', component_property='value'),
     State(component_id='dt_min_samples_split', component_property='value'),
     State(component_id='dt_splitter', component_property='value'),
     State(component_id='dt_max_depth', component_property='value'),
     State(component_id='knn_algorithm', component_property='value'),
     State(component_id='knn_neighbours', component_property='value'),
     State(component_id='knn_weight', component_property='value'),
     State(component_id='knn_distance', component_property='value'),
     State(component_id='gb_criterion', component_property='value'),
     State(component_id='gb_loss_function', component_property='value'),
     State(component_id='gb_learning_rate', component_property='value'),
     State(component_id='gb_max_depth', component_property='value'),
     State(component_id='gb_min_samples_split', component_property='value'),
     State(component_id='nb_algorithm', component_property='value'),
     State(component_id='alpha_value', component_property='value'),
     State(component_id='fit_prior', component_property='value'),
     State(component_id='toggle_class_prior', component_property='value'),
     State(component_id='slider1', component_property='value'),
     State(component_id='slider2', component_property='value'),
     State(component_id='toggle_transfer_learning', component_property='value'),
     State(component_id='model_location', component_property='value'), ],
    [Input("export-pdf", "n_clicks"), Input("show-results", "children")],

)
def update_output(type_data, algorithms, textual_select, feature_to_vectorize, numerical_select, slider_val, label,
                  contents, filename, lr_solver, lr_c,
                  lr_penalty, lr_max_iter, dt_criterion, dt_min_samples_split, dt_splitter, dt_max_depth, knn_algorithm,
                  knn_neighbours, knn_weight, knn_distance,
                  gb_criterion, gb_loss_function, gb_learning_rate, gb_max_depth, gb_min_samples_split, nb_algorithm,
                  alpha_value, fit_prior, toggle_class_prior, slider1, slider2, toggle_transfer_learning, model_location, asd, children):
    df = pd.DataFrame()
    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_data(contents, filename)
    today = date.today()

    # write the results to a pdf file
    def write_pdf(bytes_io):
        pdf = create_pdf(asd)  # pass argument to PDF creation here
        pdf.cell(40, 7, "Completion date: {}.".format(
            today.strftime("%B %d, %Y")), 0, 2)
        pdf.cell(40, 7, "Dataset file name: {}.".format(filename), 0, 2)
        pdf.cell(40, 7, "Number of rows is {} and number of columns is {}.".format(
            len(df), len(df.columns)), 0, 2)
        pdf.cell(40, 7, "List of columns in dataset: {}.".format(
            ', '.join(df.columns)), 0, 2)
        pdf.cell(40, 7, "Test and train split. Test: {}% and Train: {}%.".format(
            (100 - int(slider_val)), slider_val), 0, 2)
        if type_data == "Textual Data":
            pdf.cell(40, 7, "Vectorizer algorithm used: {}.".format(
                textual_select), 0, 2)
        else:
            pdf.cell(
                40, 7, "p-value used for most sgnificant features: {}.".format(numerical_select), 0, 2)
        pdf.ln(8)
        pdf.cell(40, 10, "List of algorithm parameters:", 0, 2)
        pdf.ln(3)
        for alg in algorithms:
            if "Logistic Regression" in alg:
                pdf.cell(40, 7, "Logistic Regression:", 0, 2)
                pdf.cell(40, 7, "Selected solver: {}.".format(lr_solver), 0, 2)
                pdf.cell(40, 7, "Regularization parameter: {}.".format(lr_c), 0, 2)
                pdf.cell(40, 7, "Penalty norm: {}.".format(lr_penalty), 0, 2)
                pdf.cell(40, 7, "Number of iterrations: {}.".format(lr_max_iter), 0, 2)
                pdf.ln(5)

            if "Decision Trees" in alg:
                pdf.cell(40, 7, "Decision Trees:", 0, 2)
                pdf.cell(40, 7, "Selected criterion: {}.".format(dt_criterion), 0, 2)
                pdf.cell(40, 7, "Selected minimum split: {}.".format(dt_min_samples_split), 0, 2)
                pdf.cell(40, 7, "Split strategy: {}.".format(dt_splitter), 0, 2)
                pdf.cell(40, 7, "Chosen maximum depth: {}.".format(dt_max_depth), 0, 2)
                pdf.ln(5)

            if "Nearest Neighbour" in alg:
                pdf.cell(40, 7, "Nearest Neighbour:", 0, 2)
                pdf.cell(40, 7, "Selected algorithm: {}.".format(knn_algorithm), 0, 2)
                pdf.cell(40, 7, "Number of neighbours: {}.".format(knn_neighbours), 0, 2)
                pdf.cell(40, 7, "Weight function: {}.".format(knn_weight), 0, 2)
                if knn_distance is '1':
                    pdf.cell(40, 7, "Distance calculation method: Manhattan distance.", 0, 2)
                else:
                    pdf.cell(40, 7, "Distance calculation method: Euclidian distance.", 0, 2)
                pdf.ln(5)

            if "Gradient Boosting" in alg:
                pdf.cell(40, 7, "Gradient Boosting:", 0, 2)
                pdf.cell(40, 7, "Selected criterion: {}.".format(gb_criterion), 0, 2)
                pdf.cell(40, 7, "Selected loss function: {}.".format(gb_loss_function), 0, 2)
                pdf.cell(40, 7, "Learning rate: {}.".format(gb_learning_rate), 0, 2)
                pdf.cell(40, 7, "Chosen maximum depth: {}.".format(gb_max_depth), 0, 2)
                pdf.cell(40, 7, "Selected minimum split: {}.".format(gb_min_samples_split), 0, 2)
                pdf.ln(5)
        pdf.ln(6)
        pdf.cell(40, 10, "List of used algorithms and their results:", 0, 2)
        pdf.ln(3)
        for child in children:
            row = []
            pdf.cell(40, 5, "{}:".format(
                child["props"]["children"][0]["props"]["children"]), 0, 2)
            for i in range(1, 5):
                row.append(child["props"]["children"][i]["props"]["children"])
                row.append("\n")
            pdf.cell(40, 5, ' '.join(row), 0, 2)
            pdf.ln(5)
        pdf.set_y(265)
        pdf.set_font('Arial', 'I', 8)
        pdf.cell(
            0, 10, 'For any questions, contact the author on: bedircaushi1@gmail.com. All rights reserved.', 0, 1, 'R')
        bytes_io.write(pdf.output(dest='S').encode('latin-1'))

    return dcc.send_bytes(write_pdf, "results.pdf")


# main callback that reads all inputs and generates the results
@app.callback(
    [Output(component_id='export-pdf-div', component_property='style'),
     Output('show-results', 'children')],
    [State(component_id='select_type_data', component_property='value'),
     State(component_id='select_models', component_property='value'),
     State(component_id='textual_selection', component_property='value'),
     State(component_id='select_feature', component_property='value'),
     State(component_id='numerical_selection', component_property='value'),
     State(component_id='slider', component_property='value'),
     State(component_id='select_label', component_property='value'),
     State("upload-data", "contents"), State("upload-data", "filename"),
     State(component_id='lr_solver', component_property='value'),
     State(component_id='lr_c', component_property='value'),
     State(component_id='lr_penalty', component_property='value'),
     State(component_id='lr_max_iter', component_property='value'),
     State(component_id='dt_criterion', component_property='value'),
     State(component_id='dt_min_samples_split', component_property='value'),
     State(component_id='dt_splitter', component_property='value'),
     State(component_id='dt_max_depth', component_property='value'),
     State(component_id='knn_algorithm', component_property='value'),
     State(component_id='knn_neighbours', component_property='value'),
     State(component_id='knn_weight', component_property='value'),
     State(component_id='knn_distance', component_property='value'),
     State(component_id='gb_criterion', component_property='value'),
     State(component_id='gb_loss_function', component_property='value'),
     State(component_id='gb_learning_rate', component_property='value'),
     State(component_id='gb_max_depth', component_property='value'),
     State(component_id='gb_min_samples_split', component_property='value'),
     State(component_id='nb_algorithm', component_property='value'),
     State(component_id='alpha_value', component_property='value'),
     State(component_id='fit_prior', component_property='value'),
     State(component_id='toggle_class_prior', component_property='value'),
     State(component_id='slider1', component_property='value'),
     State(component_id='slider2', component_property='value'),
     State(component_id='toggle_transfer_learning', component_property='value'),
     State(component_id='model_location', component_property='value'), ],
    Input("submit-val", "n_clicks"),
)
def update_output(type_data, algorithms, textual_select, feature_to_vectorize, numerical_select, slider_val, label,
                  contents, filename, lr_solver, lr_c,
                  lr_penalty, lr_max_iter, dt_criterion, dt_min_samples_split, dt_splitter, dt_max_depth, knn_algorithm,
                  knn_neighbours, knn_weight, knn_distance,
                  gb_criterion, gb_loss_function, gb_learning_rate, gb_max_depth, gb_min_samples_split, nb_algorithm,
                  alpha_value, fit_prior, toggle_class_prior, slider1, slider2, toggle_transfer_learning, model_location, asd):
    print("alg,", algorithms)
    print("textual", textual_select)
    print("numerical", numerical_select)
    print("feature to vectorize select", feature_to_vectorize)
    print("slider", slider_val)
    print("label", label)
    print(filename[0])
    print(lr_solver, lr_c,
          lr_penalty, lr_max_iter, dt_criterion, dt_min_samples_split, dt_splitter, dt_max_depth, knn_algorithm,
          knn_neighbours, knn_weight, type(knn_distance),
          gb_criterion, gb_loss_function, gb_learning_rate, gb_max_depth, gb_min_samples_split, nb_algorithm,
          alpha_value, fit_prior, toggle_class_prior, toggle_transfer_learning, model_location)

    df = pd.DataFrame()

    if contents:
        contents = contents[0]
        dataset_filename = filename[0]
        df = parse_data(contents, dataset_filename)
    # data preprocessing
    if label is not None and feature_to_vectorize is not None:
        df = drop_na(df, [feature_to_vectorize])

        df = keep_columns(df, [feature_to_vectorize, label])

        df = shuffle_csv(df)

        start_time = time.time()

        df = remove_punctuation_stopwords(df, [feature_to_vectorize])
        print("--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        df = word_stemming(df, [feature_to_vectorize])
        print("--- %s seconds ---" % (time.time() - start_time))
    # word embedding if tf idf is selected
    results = []
    if type_data == "Textual Data" and textual_select == "Tfidf":
        x = df[feature_to_vectorize]
        # x=df.drop(columns="labels",axis=1)
        y = df[label]

        # reducedX, sigFeatures = getDatasetWithSignificantFeatures(x,y,0.01)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=(100 - slider_val) / 100)

        if toggle_transfer_learning == "No":
            vectorization = TfidfVectorizer()
            xv_train = vectorization.fit_transform(x_train)
            xv_test = vectorization.transform(x_test)
            pickle.dump(vectorization, open("vectorizer.sav", 'wb'))
        else:
            vectorization = pickle.load(open("vectorizer.sav", 'rb'))
            xv_train = vectorization.transform(x_train)
            xv_test = vectorization.transform(x_test)

    # word embedding if hybrid model is selected

    if type_data == "Textual Data" and textual_select == "Hybrid Model":
        df.to_csv("data_for_feature_extraction.csv")
        feature_df = feature_extraction(
            "data_for_feature_extraction.csv", feature_to_vectorize, label)
        vectorize_df = vectorize_feature(
            "data_for_feature_extraction.csv", feature_to_vectorize)

        merged = mergeDatasetWithLabels(feature_df, vectorize_df)

        x = merged.drop(columns="labels", axis=1)
        y = merged["labels"]

        reducedX, sigFeatures = getDatasetWithSignificantFeatures(x, y, 0.05)
        xv_train, xv_test, y_train, y_test = train_test_split(
            reducedX, y, test_size=(100 - slider_val) / 100)

    # word embedding if feature extraction is selected

    if type_data == "Textual Data" and textual_select == "Feature Extraction":
        df.to_csv("data_for_feature_extraction.csv")

        feature_df = feature_extraction(
            "data_for_feature_extraction.csv", feature_to_vectorize, label)

        x = feature_df.drop(columns='labels', axis=1)
        y = feature_df['labels']

        reducedX, sigFeatures = getDatasetWithSignificantFeatures(x, y, 0.05)
        xv_train, xv_test, y_train, y_test = train_test_split(
            reducedX, y, test_size=(100 - slider_val) / 100)

    # following code block determines which algorithm is selceted, builds machine learing model, extraxts it and generates results
    final_div = []
    for alg in algorithms:
        if "Logistic Regression" in alg:
            LR = LogisticRegression(
                solver=lr_solver, C=lr_c, penalty=lr_penalty, max_iter=lr_max_iter)
            LR.fit(xv_train, y_train)

            filename = "pickles/" + \
                       alg.replace(" ", "") + "-" + dataset_filename.split(".")[0] + ".sav"
            pickle.dump(LR, open(filename, "wb"))

            pred_lr = LR.predict(xv_test)
            score = LR.score(xv_test, y_test)

            results.append(classification_report(
                y_test, pred_lr, output_dict=True))

            final_div.append(get_results(alg, precision_score(y_test, pred_lr), recall_score(
                y_test, pred_lr), score, f1_score(y_test, pred_lr)))

        if "Decision Trees" in alg:
            DT = DecisionTreeClassifier(
                criterion=dt_criterion, min_samples_split=dt_min_samples_split, splitter=dt_splitter,
                max_depth=dt_max_depth)
            DT.fit(xv_train, y_train)

            filename = "pickles/" + \
                       alg.replace(" ", "") + "-" + dataset_filename.split(".")[0] + ".sav"
            pickle.dump(DT, open(filename, "wb"))

            pred_dt = DT.predict(xv_test)
            score = DT.score(xv_test, y_test)

            results.append(classification_report(
                y_test, pred_dt, output_dict=True))

            final_div.append(get_results(alg, precision_score(y_test, pred_dt), recall_score(
                y_test, pred_dt), score, f1_score(y_test, pred_dt)))

        if "Nearest Neighbour" in alg:
            knn = KNeighborsClassifier(
                algorithm=knn_algorithm, n_neighbors=knn_neighbours, weights=knn_weight, p=int(knn_distance))
            knn.fit(xv_train, y_train)

            filename = "pickles/" + \
                       alg.replace(" ", "") + "-" + dataset_filename.split(".")[0] + ".sav"
            pickle.dump(knn, open(filename, "wb"))

            pred_knn = knn.predict(xv_test)
            score = knn.score(xv_test, y_test)

            results.append(classification_report(
                y_test, pred_knn, output_dict=True))

            final_div.append(get_results(alg, precision_score(y_test, pred_knn), recall_score(
                y_test, pred_knn), score, f1_score(y_test, pred_knn)))

        if "Gradient Boosting" in alg:
            GBC = GradientBoostingClassifier(criterion=gb_criterion, loss=gb_loss_function,
                                             learning_rate=gb_learning_rate, max_depth=gb_max_depth,
                                             min_samples_split=gb_min_samples_split)
            GBC.fit(xv_train, y_train)

            filename = "pickles/" + \
                       alg.replace(" ", "") + "-" + dataset_filename.split(".")[0] + ".sav"
            pickle.dump(GBC, open(filename, "wb"))

            pred_gbc = GBC.predict(xv_test)
            score = GBC.score(xv_test, y_test)

            results.append(classification_report(
                y_test, pred_gbc, output_dict=True))

            final_div.append(get_results(alg, precision_score(y_test, pred_gbc), recall_score(
                y_test, pred_gbc), score, f1_score(y_test, pred_gbc)))

        if "Naive Bayes" in alg:
            fit_p = True if fit_prior == "Enabled" else False
            temp_class_prior = [slider1, slider2]
            class_p = None if not toggle_class_prior else temp_class_prior
            if nb_algorithm == "Bernoulli":
                if toggle_transfer_learning == "No":
                    BNB = BernoulliNB(alpha=alpha_value, fit_prior=fit_p, class_prior=class_p)
                    BNB.fit(xv_train, y_train)

                    filename = "pickles/" + \
                               alg.replace(" ", "") + "-" + dataset_filename.split(".")[0] + ".sav"

                    pickle.dump(BNB, open(filename, "wb"))

                else:
                    BNB = pickle.load(open(model_location, "rb"))
                    BNB.partial_fit(xv_train, y_train, classes=np.unique(y_train))

                pred_bnb = BNB.predict(xv_test)
                score = BNB.score(xv_test, y_test)

                results.append(classification_report(
                    y_test, pred_bnb, output_dict=True))
                final_div.append(get_results(alg, precision_score(y_test, pred_bnb), recall_score(
                    y_test, pred_bnb), score, f1_score(y_test, pred_bnb)))
            else:
                if toggle_transfer_learning == "No":
                    MNB = MultinomialNB(alpha=alpha_value, fit_prior=fit_p, class_prior=class_p)
                    MNB.fit(xv_train, y_train)

                    filename = "pickles/" + \
                                alg.replace(" ", "") + "-" + dataset_filename.split(".")[0] + ".sav"

                    pickle.dump(MNB, open(filename, "wb"))
                else:
                    MNB = pickle.load(open(model_location, "rb"))
                    MNB.partial_fit(xv_train, y_train, classes=np.unique(y_train))

                pred_mnb = MNB.predict(xv_test)
                score = MNB.score(xv_test, y_test)

                results.append(classification_report(
                    y_test, pred_mnb, output_dict=True))
                final_div.append(get_results(alg, precision_score(y_test, pred_mnb), recall_score(
                    y_test, pred_mnb), score, f1_score(y_test, pred_mnb)))

    return ({'display': 'block'}, final_div)


# get the list of features from the uploaded dataset
@app.callback(
    Output("select_label", "options"),
    [Input("upload-data", "contents"), Input("upload-data", "filename")],
)
def get_feature_list(contents, filename):
    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_data(contents, filename)
        column_names = list(df.columns.values)
        return [{'label': x, 'value': x} for x in column_names]

    return []


# dropdown to select the feature to vectorize
@app.callback(
    Output("select_feature", "options"),
    [Input("upload-data", "contents"), Input("upload-data", "filename")],
)
def get_feature_list(contents, filename):
    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_data(contents, filename)
        column_names = list(df.columns.values)
        return [{'label': x, 'value': x} for x in column_names]

    return []


if __name__ == "__main__":
    app.run_server(debug=True, threaded=False)
