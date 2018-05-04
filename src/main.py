import os
import csv
import sklearn
import numpy as np
import statistics as st
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import train_test_split

# from setup import setup

class Dataset(object):
    """The Dataset class is specific to the crop yield dataset collected for the
    purpose of analysis of Computer Vision techniques, and the development of a
    final project.

    The resultant dataset supports two primary research interests for the projcet:
        1)  Crop yield measurements, with associated multispectral data collected
                from the LandSat8 satelite
        2)  Crop classifications associated with the same multispectral data.

    This is to support both Classification and Regression techniques associated
    with support vectors and Random Forrest Classifiers.

    Contained Variables:
        .data           --  Numpy array of records, each containing 10 features
                            (found in .feature_names)
        .target_yield   --  Numpy array of dry yields (measured in bu/ac), for
                            the crops (found in .target_names) associated with .data
        .target_crop    --  Numpy array of crop classification associated with .data
        .feature_names  --  Description of all the fields found in each record
        .target_names   --  Crop types covered in this dataset, correspondant to
                            index value

    Suppoted Functions:
        .corn()         --  This will return a Dataset class object containing
                            only the subset of records measuring corn fields
        .soybeans()     --  This will return a Dataset class object containing
                            only the subset of records measuring soybean fields
    """
    def __init__(self, data_list, target_yield, target_crop, data_features_list, target_names):
        self.data = np.array(data_list)
        self.target_yield = np.array(target_yield)
        tc = np.array(target_crop)
        self.target_crop = tc.astype(int)
        self.feature_names = data_features_list
        self.target_names = target_names

    def corn(self):
        data_list = list()
        target_yield = list()
        target_crop = list()

        for i in range(0, len(self.data)):
            if not self.target_crop[i]:
                data_list.append(self.data[i])
                target_yield.append(self.target_yield[i])
                target_crop.append(self.target_crop[i])

        data_features_list = self.feature_names
        target_names = ['CORN']

        return Dataset(data_list, target_yield, target_crop, data_features_list, target_names)

    def soybeans(self):
        data_list = list()
        target_yield = list()
        target_crop = list()

        for i in range(0, len(self.data)):
            if self.target_crop[i]:
                data_list.append(self.data[i])
                target_yield.append(self.target_yield[i])
                target_crop.append(self.target_crop[i])

        data_features_list = self.feature_names
        target_names = ['SOYBEANS']

        return Dataset(data_list, target_yield, target_crop, data_features_list, target_names)


def file_path():
    return os.getcwd()

def file_list(PATH):
    return os.listdir(PATH)

def normal(DATA):
    d = [i['Count'] for i in DATA]
    return st.median(d), st.stdev(d)

def load_data():
    """ Main default function that builds and returns the dataset in a Dataset
    class object.
    """
    FEATURE_DICT = {
        'Count':   'Count',
        'Product': 'Crop',
        'NDVI':    'NDVI',
        'NDWI2':   'NDWI (McFeeters)',
        'NDWI1':   'NDWI (Gao)',
        'G':       'Green',
        'R':       'Red',
        'NIR':     'Near IR',
        'SIR1':    'Shortwave IR 1',
        'SIR2':    'Shortwave IR 2',
        'TIR1':    'Thermal IR 1',
        'TIR2':    'Thermal IR 2'
    }
    DATAPATH = "/".join(file_path().split('/')[:-1]) + "/data/proc/"
    files = file_list(DATAPATH)

    # READ ALL DATA INTO ONE LIST OF RECORDS
    raw = list()
    for f in files:
        if f.endswith('.csv'):
            with open(DATAPATH + f) as rfp:
                reader = csv.DictReader(rfp)
                for r in reader:
                    d = dict()
                    # Cleaning up the record types (they're all strings by default)
                    for k in FEATURE_DICT.keys():
                        if k == 'Product':
                            # Because strings aren't a valid feature type,
                            # we're converting corn = 0 and soybeans = 1
                            if r[k] == "['CORN']":
                                d[k] = 0
                            elif r[k] == "['SOYBEANS']":
                                d[k] = 1
                            else:
                                continue
                            # Because we cant use a string as a feature, we're
                        elif k in ['NDVI', 'NDWI1', 'NDWI2']:
                             d[k] = float(r[k])
                        else:
                            d[k] = int(r[k])
                    # Decided to drop all the floating point values in the
                    # yield. It doesn't add accuracy to a (bu/ac) measure
                    d['Yld Vol(Dry)(bu/ac)'] = round(float(r['Yld Vol(Dry)(bu/ac)']))
                    if 'Product' in d.keys():
                        raw.append(d)
                    else:
                        continue

    # BUILD DATASET PRIMATIVES
    data = list()
    target_yield = list()
    target_crop = list()

    median, stdev = normal(raw)
    low = median - (stdev * 3)
    high = median + (stdev * 3)

    FEATURE_DICT.pop('Product')
    for r in raw:
        if not (low <= r['Count'] <= high):
            continue
        record = [r[k] for k in sorted(FEATURE_DICT.keys()) if k != 'Count']
        t_yield = r['Yld Vol(Dry)(bu/ac)']
        t_crop = r['Product']
        data.append(record)
        target_yield.append(t_yield)
        target_crop.append(t_crop)

    FEATURE_DICT.pop('Count')
    feature_names = [FEATURE_DICT[k] for k in sorted(FEATURE_DICT.keys())]
    target_names = ["CORN", "SOYBEANS"]

    return Dataset(data, target_yield, target_crop, feature_names, target_names)

def std_transform(DATA):
    """Uses StandardScaler to standardize given input data, returning it back.
    """
    scaler = StandardScaler()
    scaler.fit(DATA)
    return scaler.transform(DATA)

def do_pca(DATA, LABELS=None, COMPONENTS=None):
    """Principal Component Analysis function.

    Takes a numpy array of data, and returns a transformed array according to
    the number principal components specified (by default, COMPONENTS is set to
    the total number of components in each data row).

    Arguments:
    - LABELS:       If you'd like to see the findings of the PCA, pass a list
                    of lables for each feature, and the function will print out
                    all the varience ratios for each given feature.
    - COMPONENTS:   Specific number of components you wish to consider in the PCA
                    .. will pair down the data to that number of components
    """
    nc = int()
    if COMPONENTS != None:
        nc = COMPONENTS
    else:
        nc = len(LABELS)
    pca = PCA(n_components=nc)
    p = pca.fit_transform(DATA)
    if LABELS:
        for i in range(0, pca.n_components_):
            print pca.explained_variance_ratio_[i] * 100, LABELS[i]
    return p

def random_forrest(X, Y):
    rf_class = RandomForestClassifier(n_estimators=10)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)
    rf_class.fit(X_train, y_train)
    print("Random Forests:\t\t{}".format(rf_class.score(X_test, y_test)))
    kf = KFold(len(X), n_folds=10, shuffle=True, random_state=10)
    print("Random Forest w/ KF:\t{}".format(cross_val_score(rf_class, X, Y, cv=kf, scoring='accuracy').mean()))

def svr_est(X,Y):
    svr = SVR(kernel='linear')
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)
    svr.fit(X_train, y_train)
    print("SVR:\t\t{}".format(svr.score(X_test, y_test)))
    kf = KFold(len(X), n_folds=10, shuffle=True, random_state=10)
    print("SVR w/ KF:\t{}".format(cross_val_score(svr, X, Y, cv=kf, scoring='r2').mean()))

def main():
    return load_data()

if __name__ == '__main__':
    main()
