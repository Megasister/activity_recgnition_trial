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
from sklearn_porter import Porter

# n_class = ['jog', 'others', 'rope_skipping', 'sit_up1', 'sit_up2', 'walking']
n_class = ['jog', 'others', 'rope_skipping', 'sit_up1', 'walking']
# n_class = ['others', 'rope_skipping', 'sit_up1']
# n_class = ['jog', 'others', 'walking']


# 获得合成矢量
def sumvector(*args):
    return math.sqrt(sum([pow(arg, 2) for arg in args]))


def get_train_test_data(ftlist, taglist, i, n_class, test_size, random_state, shuffle_state):
    df_ft = shuffle(pd.DataFrame(ftlist).loc[i].dropna(how='all'), random_state=shuffle_state)
    df_tags = shuffle(pd.DataFrame(taglist).loc[i].dropna(how='all'), random_state=shuffle_state)
    # except:
    #     print（
    #     df_ft = shuffle(pd.DataFrame(ftlist[i]).T, random_state=shuffle_state).T
    #     df_tags = shuffle(pd.DataFrame(taglist[i], index=[0]).T, random_state=shuffle_state)

    if len(n_class) == 2:
        df_tags = convert_multiclass_binary(df_tags, n_class, i)

    # ravel the inner part (21d array) of feature child_data matrix
    X = []
    for _, v in df_ft.items():
        temp = []
        try:
            for line in v:  # for one dimension data
                temp.extend(line.tolist())
        except:
            for line in v[0]:  # for multi dimension data
                temp.extend(np.array(line).ravel().tolist())
        X.append(temp)
    Y = df_tags.T.to_numpy().ravel().tolist()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    print(np.asarray(X_train).shape)
    print(np.asarray(X_test).shape)
    print(np.asarray(Y_train).shape)
    print(np.asarray(Y_test).shape)
    return X_train, X_test, Y_train, Y_test


def convert_multiclass_binary(df_tags, n_class, i):
    for k in range(len(df_tags)):
        if df_tags.iloc[k][0] != n_class[i]:
            df_tags.iloc[k][0] = 'others'

    return df_tags


def process_data():
    window_sizes = [0.5, 1, 3, 5, 8, 10]
    norm = 'norm'
    ad_no = []
    chld_no = [3]

    # get corresponding dataframe from adult and child groups
    content = ''
    ft_ad, tag_ad, ft_chld, tag_chld = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for ad in ad_no:
        content += '+ ad{}'.format(ad)
        ft = pd.read_pickle(r'multiclass_new_tag\feature{}_w{}_{}.pickle'.format(ad, window_sizes, norm))
        tag = pd.read_pickle(r'multiclass_new_tag\tag{}_w{}_{}.pickle'.format(ad, window_sizes, norm))
        ft_ad = pd.concat([ft_ad, ft], axis=1, sort=False)
        tag_ad = pd.concat([tag_ad, tag], axis=1, sort=False)
    for chld in chld_no:
        content += '+ ch{}'.format(chld)
        ft = pd.read_pickle(r'multiclass_new_tag\feature_child{}_w{}_{}.pickle'.format(chld, window_sizes, norm))
        tag = pd.read_pickle(r'multiclass_new_tag\tag_child{}_w{}_{}.pickle'.format(chld, window_sizes, norm))
        ft_chld = pd.concat([ft_chld, ft], axis=1, sort=False)
        tag_chld = pd.concat([tag_chld, tag], axis=1, sort=False)



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
    ftlist = pd.concat([ft_chld, ft_ad], axis=1, sort=False)
    taglist = pd.concat([tag_chld, tag_ad], axis=1, sort=False)

    # ORDER MATTERS
    window_sizes = [0.5, 1, 3, 5, 8, 10]

    for i in range(len(window_sizes)):
        X_train, X_test, Y_train, Y_test = get_train_test_data(ftlist, taglist, i, n_class, test_size=0.3,
                                                               random_state=10, shuffle_state=16)
        # clf = svm.SVC(kernel='rbf', gamma=1.0)
        # clf = OneVsRestClassifier(GaussianNB())
        # clf = MultinomialNB()
        # clf = GaussianNB()
        # clf = BernoulliNB()
        clf = RandomForestClassifier()
        # Train the model using the training sets
        clf.fit(X_train, Y_train)

        # porter = Porter(clf, language='C')
        # output = porter.export(embed_data=True)
        # print(output)
        #
        # # Save model:
        # with open('forest.c', 'w') as f:
        #     f.write(output)

        Y_predict = clf.predict(X_test)
        Y_prob = clf.predict_proba(X_test)
        print(X_test[100])
        print(Y_predict[100])

        # Y_test = label_binarize(Y_test, classes=n_class)
        # Y_predict = label_binarize(Y_predict, classes=n_class)
        # cr_sc = cross_val_score(clf, X_test , Y_test, cv=5)
        # print(cr_sc)

        plot_evaluation.plot_cf_mat(clf, X_test, Y_test, window_sizes[i], norm, content, test_ratio=0.3)
        report = classification_report(Y_test, Y_predict)
        print(report)
        if len(n_class) > 2:
            plot_evaluation.plot_roc(Y_test, Y_predict, Y_prob, n_class, window_sizes[i], norm, content, test_ratio=0.3)
        else:
            plot_roc_curve(clf, X_test, Y_test)


if __name__ == '__main__':
    # load_data.load_init_data()
    process_data()

