import base64
import io

import dash_html_components as html

import pandas as pd
import numpy as np

import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from empath import Empath
import time
from langdetect import detect

from sklearn.feature_selection import SelectKBest, f_classif

from fpdf import FPDF
from sklearn.feature_extraction.text import TfidfVectorizer

# All the methods for data preprocessing, word embedding, feature selection, machine learning algorithms and more.

nltk.download('stopwords')
nltk.download('punkt')


def parse_data(contents, filename):
    content_type, content_string = contents.split(",")

    decoded = base64.b64decode(content_string)
    try:
        if "csv" in filename:
            # Assume that the user uploaded a CSV or TXT file
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        elif "xls" in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        elif "txt" or "tsv" in filename:
            # Assume that the user upl, delimiter = r'\s+'oaded an excel file
            df = pd.read_csv(io.StringIO(
                decoded.decode("utf-8")), delimiter=r"\s+")
    except Exception as e:
        print(e)
        return html.Div(["There was an error processing this file."])
    return df


def create_pdf(n_nlicks):
    if (n_nlicks != 0):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 18)
        # Move to the right
        # Title
        pdf.cell(45, 10, 'Results Report', 0, 1, 'C')
        # Line break
        pdf.ln(20)
        pdf.set_font('Arial', '', 11)

        return pdf
    return 0


def get_results(algorithm, precision, recall, accuracy, f1_score):
    table = html.Div(
        [
            html.H5("{}".format(algorithm)),
            html.B("ACCURACY: {0:.0%} | ".format(accuracy)),
            html.Span("Precision: {0:.0%} | ".format(precision)),
            html.Span("Recall: {0:.0%} | ".format(recall)),
            html.Span("F1score: {0:.0%} | ".format(f1_score)),
            html.Hr()
        ]
    )
    return table


def feature_extraction(input, column, label):
    start_time = time.time()
    lexicon = Empath()

    res = lexicon.analyze("po", normalize=True)
    dict_keys = []
    for key in res:
        dict_keys.append(key)

    df = pd.DataFrame(columns=dict_keys)
    df1 = pd.read_csv(input)
    dict = []
    for vals in df1[column]:
        res = lexicon.analyze(vals, normalize=True)
        dict.append(res)

    df = pd.DataFrame(dict)
    df = df.assign(labels=df1[label])
    print("--- %s seconds ---" % (time.time() - start_time))
    return df


def computeAnova(dataset, labels):
    # scikit learn library method to perform ANOVA
    fstatAnova = SelectKBest(score_func=f_classif, k='all')
    # here computed dataframe returns the same dataset but with information about
    #f-scores and p-values
    fstatAnova.fit(dataset, labels)
    return fstatAnova


def getDatasetWithSignificantFeatures(dataset, labels, pvalueMargin):
    fanova = computeAnova(dataset, labels)
    # variables to store values and indexes of every iterration
    notsignificantFeatureIndex = 0
    colIndices = list()
    significantFeatures = list()
    notSignificantFeatures = list()
    j = 1
    # loop to iterrate on every feature of dataset
    for i in range(len(fanova.scores_)):
        # condition to take important features based on pvalue and write the new dataset
        if fanova.pvalues_[i] < pvalueMargin:
            j += 1
            significantFeatures.append(fanova.scores_[i])
        else:
            notsignificantFeatureIndex += 1
            colIndices.append(i)
            notSignificantFeatures.append(fanova.scores_[i])

    # new reduced dataset with significant features
    df = pd.DataFrame(dataset)
    reducedDataSet = df.drop(df.columns[colIndices], axis=1)

    return reducedDataSet, significantFeatures


def drop_na(input, columns):
    df = input
    for column in columns:
        df[column].replace('', np.nan, inplace=True)
        df.dropna(subset=[column], inplace=True)
        df = df[df[column].str.split().str.len().gt(50)]
        df['detect'] = df[column].apply(detect)
        df = df[df['detect'] == 'en']

    return df


def drop_non_english(input, columns):
    df = input
    for column in columns:
        df['detect'] = df[column].apply(detect)
        df = df[df['detect'] == 'en']

    return df


def shuffle_csv(input):
    df = input
    shuffled_df = df.sample(frac=1)

    return shuffled_df


def remove_punctuation_stopwords(input, columns):
    df = input
    stop = stopwords.words('english')
    for column in columns:
        df[column].replace('', np.nan, inplace=True)
        df.dropna(subset=[column], inplace=True)
        df[column] = df[column].apply(
            lambda x: " ".join(x.lower() for x in x.split()))
        df[column] = df[column].str.replace('[^\w\s]', '')

        df[column] = df[column].apply(lambda x: " ".join(
            x for x in x.split() if x not in stop))

    return df


def word_stemming(input, columns):
    df = input
    stemmer = PorterStemmer()

    for column in columns:
        df[column].replace('', np.nan, inplace=True)
        df.dropna(subset=[column], inplace=True)
        df[column] = df[column].apply(lambda row: word_tokenize(row))
        df[column] = df[column].apply(
            lambda row: [stemmer.stem(y) for y in row])
        df[column] = df[column].apply(lambda row: " ".join(row))

    return df


def add_label(input, label):
    csv_input = pd.read_csv(input)
    csv_input["label"] = label

    return csv_input


def keep_columns(input, attributes):
    f = input
    keep_col = attributes
    new_f = f[keep_col]

    return new_f


def mergeDatasetWithLabels(dataset, labels):
    _dataset = dataset
    _labels = labels
    _labels = _labels.dropna(axis=1)
    return pd.merge(_dataset, _labels, left_index=True, right_index=True, how="outer")
    # merged.to_csv(output, index=False)


def vectorize_feature(input, column):
    dataset = pd.read_csv(input)
    x = dataset[column]
    tfidf_vectorizer = TfidfVectorizer(
        min_df=1,  # min count for relevant vocabulary
        strip_accents='unicode',  # replace all accented unicode char
        # by their corresponding  ASCII char
        analyzer='word',  # features made of words
        token_pattern=r'\w{1,}',  # tokenize only words of 4+ chars
        ngram_range=(1, 1),  # features made of a single tokens
        use_idf=True,  # enable inverse-document-frequency reweighting
        smooth_idf=True,  # prevents zero division for unseen words
        sublinear_tf=False)

    tfidf_df = tfidf_vectorizer.fit_transform(x)

    names = tfidf_vectorizer.get_feature_names()
    data = tfidf_df.todense().tolist()

    return pd.DataFrame(data, columns=names)
