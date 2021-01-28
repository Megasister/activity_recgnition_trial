import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import parameter
import scipy.io
import seaborn as sns
import re
import math
from feature_extraction.tsfeature import feature_core, feature_fft, feature_time
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.metrics import SCORERS, confusion_matrix, \
     plot_confusion_matrix,  plot_roc_curve, roc_curve, \
     roc_auc_score, f1_score, auc, classification_report
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from dtw import distance, dtw
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.preprocessing import LabelEncoder as le, OneHotEncoder as onehot
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy import signal
import txt_preprocess
import plot_evaluation
import feature_eng
import quadranion
import get_feature
import load_data

# n_class = ['jog', 'others', 'rope_skipping', 'sit_up1', 'sit_up2', 'walking']
axises = ['xa', 'ya', 'za', 'xw', 'yw', 'zw']


# 获得合成矢量
def sumvector(*args):
    return math.sqrt(sum([pow(arg, 2) for arg in args]))


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


def feature_norm(df_ft, reg=None):
    '''
    convert an entangled intake dataframe into a flattened list
    trained list only contains #features(66) * #samples
    features on 6 axis are flattened
    '''
    print(df_ft)
    feature_sample = df_ft.iloc[0][0]
    total = []
    for axis_ord in range(len(axises)):
        for i in range(len(feature_sample)):
            each_feature = []
            for sample_no in range(len(df_ft)):
                each_feature.append(df_ft.iloc[sample_no][axis_ord][i])
            total.append(each_feature)

    '''
    total dimension: 66(#axis * #features on each axis) * #samples
    order does not change
    '''
    new_bundle = []
    for ele in total:
        if reg == None:
            ft_vec = ele
        if reg == 'norm':
            ft_vec = normalization(ele)
        elif reg == 'stdr':
            ft_vec = standardization(ele)
        new_bundle.append(ft_vec)
    return np.array(new_bundle).T.tolist()


def get_train_test_data(ftlist, taglist, i, para, test_size, random_state, shuffle_state):
    df_ft = shuffle(pd.DataFrame(ftlist).loc[i].dropna(how='all'), random_state=shuffle_state)
    df_tags = shuffle(pd.DataFrame(taglist).loc[i].dropna(how='all'), random_state=shuffle_state)

    X = feature_norm(df_ft, para['feature_norm'])
    Y = df_tags.T.to_numpy().ravel().tolist()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    print(np.asarray(X_train).shape)
    print(np.asarray(X_test).shape)
    print(np.asarray(Y_train).shape)
    print(np.asarray(Y_test).shape)
    # except:
    #     print（
    #     df_ft = shuffle(pd.DataFrame(ftlist[i]).T, random_state=shuffle_state).T
    #     df_tags = shuffle(pd.DataFrame(taglist[i], index=[0]).T, random_state=shuffle_state)

    # if len(n_class) == 2:
    #     df_tags = convert_multiclass_binary(df_tags, n_class, i)

    # ravel the inner part (21d array) of feature child_data matrix
    # X = []
    # for _, v in df_ft.items():
    #     temp = []
    #     try:
    #         for line in v:  # for one dimension data
    #             temp.extend(line.tolist())
    #     except:
    #         for line in v[0]:  # for multi dimension data
    #             temp.extend(np.array(line).ravel().tolist())
    #     X.append(temp)
    return X_train, X_test, Y_train, Y_test


def convert_multiclass_binary(df_tags, n_class, i):
    for k in range(len(df_tags)):
        if df_tags.iloc[k][0] != n_class[i]:
            df_tags.iloc[k][0] = 'others'

    return df_tags


def process_data():
    # # take data and process
    # window_sizes = [0.5, 1, 3, 5, 8, 10]
    window_sizes = [4.92]
    data_norm = None  # there is global option and local (within sliding window) option
    feature_norm = None
    ad_no = [1, 2]
    chld_no = [1, 2, 3, 5]

    # get corresponding dataframe from adult and child groups
    content = ''
    ft_ad, tag_ad, ft_chld, tag_chld = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for ad in ad_no:
        content += '+ ad{}'.format(ad)
        ft = pd.read_pickle(r'multiclass_new_tag\feature{}_w{}_{}.pickle'.format(ad, window_sizes, data_norm))
        tag = pd.read_pickle(r'multiclass_new_tag\tag{}_w{}_{}.pickle'.format(ad, window_sizes, data_norm))
        ft_ad = pd.concat([ft_ad, ft], axis=1, sort=False)
        tag_ad = pd.concat([tag_ad, tag], axis=1, sort=False)
    for chld in chld_no:
        content += '+ ch{}'.format(chld)
        ft = pd.read_pickle(r'multiclass_new_tag\feature_child{}_w{}_{}.pickle'.format(chld, window_sizes, data_norm))
        tag = pd.read_pickle(r'multiclass_new_tag\tag_child{}_w{}_{}.pickle'.format(chld, window_sizes, data_norm))
        ft_chld = pd.concat([ft_chld, ft], axis=1, sort=False)
        tag_chld = pd.concat([tag_chld, tag], axis=1, sort=False)

    # ftlist = [ftlist]
    # taglist = [taglist]
    # ftlist = ft1
    # taglist = tag1

    # print('ft1',ftlist.iloc[0].iloc[0])
    # print('ft1',ft1.iloc[0].iloc[0])
    # for mono dataset only, do not combine dataset
    # feature_eng.feature_corr(ft1, tag1, n_class, cross='feature', corr='spearman')
    # plt.show()
    # get correlation heatmap of features vs acts and features vs features
    # ft1 = pd.read_pickle(r'multiclass_new_tag\feature1_w10_norm.pickle')
    # tag1 = pd.read_pickle(r'multiclass_new_tag\tag1_w10_norm.pickle')
    # ft1 = pd.read_pickle(r'multiclass_new_tag\feature1_w[0.5, 1, 3, 5, 8, 10]_None.pickle')
    # tag1 = pd.read_pickle(r'multiclass_new_tag\tag1_w[0.5,1,3,5,8,10].pickle')
    # ft2 = pd.read_pickle(r'multiclass_new_tag\feature_child2_w10_norm.pickle')
    # tag2 = pd.read_pickle(r'multiclass_new_tag\tag_child2_w10_norm.pickle')
    # ft3 = pd.read_pickle(r'multiclass_new_tag\feature_child3_w10_norm.pickle')
    # tag3 = pd.read_pickle(r'multiclass_new_tag\tag_child3_w10_norm.pickle')
    # ft4 = pd.read_pickle(r'multiclass_new_tag\feature1_w5.pickle')
    # tag4 = pd.read_pickle(r'multiclass_new_tag\tag1_w5.pickle')
    # ft5 = pd.read_pickle(r'multiclass_new_tag\feature2_w5.pickle')
    # tag5 = pd.read_pickle(r'multiclass_new_tag\tag2_w5.pickle')

    n_class = ['jog', 'others', 'rope_skipping', 'sit_up1', 'walking']

    for i in range(len(window_sizes)):

        para = {'data_norm': data_norm, 'content': content,
                'window_size': window_sizes[i], 'n_class': n_class, 'feature_norm': feature_norm}

        # if window_sizes[i] != 5: continue
        X_train1, X_test1, Y_train1, Y_test1, X_train2, X_test2, Y_train2, Y_test2 = [], [], [], [], [], [], [], []

        if not tag_ad.empty:
            X_train1, X_test1, Y_train1, Y_test1 = get_train_test_data(ft_ad, tag_ad, i, para, test_size=0.3,
                                                   random_state=8, shuffle_state=26)
        if not tag_chld.empty:
            X_train2, X_test2, Y_train2, Y_test2 = get_train_test_data(ft_chld, tag_chld, i, para, test_size=0.3,
                                                   random_state=8, shuffle_state=26)

        Y_train1 = ['sit_up1' if a == 'sit_up2' else a for a in Y_train1]
        Y_train2 = ['sit_up1' if a == 'sit_up2' else a for a in Y_train2]
        Y_test1 = ['sit_up1' if a == 'sit_up2' else a for a in Y_test1]
        Y_test2 = ['sit_up1' if a == 'sit_up2' else a for a in Y_test2]

        # single_class_str = 'rope_skipping'
        # Y_train1 = [a if a == single_class_str else 'others' for a in Y_train1]
        # Y_train2 = [a if a == single_class_str else 'others' for a in Y_train2]
        # Y_test1 = [a if a == single_class_str else 'others' for a in Y_test1]
        # Y_test2 = [a if a == single_class_str else 'others' for a in Y_test2]
        # para['n_class'] = [single_class_str, 'others']

        X_train = shuffle(X_train1 + X_train2, random_state=10)
        Y_train = shuffle(Y_train1 + Y_train2, random_state=10)
        X_test = shuffle(X_test1 + X_test2, random_state=9)
        Y_test = shuffle(Y_test1 + Y_test2, random_state=9)

        print('Train sample size:', len(Y_train))
        print('Test sample size:', len(Y_test))
        test_ratio = len(Y_test)/(len(Y_test)+len(Y_train))
        print('Test size:', test_ratio)
        para['test_ratio'] = test_ratio

        # clf = svm.SVC(kernel='rbf', gamma=1.0)
        # clf = OneVsRestClassifier(GaussianNB())
        # clf = GaussianNB()
        # clf = BernoulliNB()
        clf = RandomForestClassifier()
        para['clf_name'] = clf.__class__.__name__
        # parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
        # clf = svm.SVC()
        # clf = GridSearchCV(svc, parameters, cv=5)

        # Train the model using the training sets
        clf.fit(X_train, Y_train)
        Y_predict = clf.predict(X_test)
        Y_prob = clf.predict_proba(X_test)

        # Y_test = label_binarize(Y_test, classes=n_class)
        # Y_predict = label_binarize(Y_predict, classes=n_class)

        plot_evaluation.plot_cf_mat(clf, X_test, Y_test, para)
        report = classification_report(Y_test, Y_predict)
        print(report)
        cr_sc = cross_val_score(clf, X_test, Y_test, cv=10)
        print(cr_sc)
        if len(para['n_class']) > 2:
            plot_evaluation.plot_roc(Y_test, Y_predict, Y_prob, para)
        else:
            plot_roc_curve(clf, X_test, Y_test)
            plt.title('ROC curve by {} for {}, test_ratio={:.3} \n'
                      'window_size={}, with {} preprocessing and {} feature norm'.format(para['clf_name'],
                                                                                         para['content'],
                                                                                         para['test_ratio'],
                                                                                         para['window_size'],
                                                                                         para['data_norm'],
                                                                                         para['feature_norm']))
            plt.show()


if __name__ == '__main__':
    # load_data.load_init_data()
    process_data()