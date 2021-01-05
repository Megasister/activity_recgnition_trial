import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, kendalltau
import dtw
from matplotlib import pylab as plt


# pearson：Pearson相关系数来衡量两个数据集合是否在一条线上面，即针对线性数据的相关系数计算，针对非线性数据便会有误差。
# spearman：非线性的，非正太分析的数据的相关系数
# kendall：用于反映分类变量相关性的指标，即针对无序序列的相关系数，非正太分布的数据
def corrolate(list1, list2, corr='dtw'):
    # get the front overlapping slice
    min_len = min(len(list1), len(list2))
    list1, list2 = list1[:min_len], list2[:min_len]

    if corr == 'spearman':
        res, _ = spearmanr(list1, list2)
    elif corr == 'pearson':
        res, _ = pearsonr(list1, list2)
    elif corr == 'kendall':
        res, _ = kendalltau(list1, list2)
    else:
        res = dtw.dtw(list1, list2)
    return res


def feature_corr(ftlist, taglist, n_class, cross='feature', corr='dtw'):
    ftlist = ftlist.T[0].apply(pd.Series).T
    ftrs = {'xa':[], 'ya':[], 'za':[], 'xw':[], 'yw':[], 'zw':[]}
    for i, k in enumerate(ftrs.keys()):
        ftrs[k] = ftlist.iloc[i]

    # only do one axis (zw) for sake of analysis across activity
    tag_class, feature = {a:[] for a in n_class}, {a:[] for a in n_class}

    for i in taglist.keys():
        tag_class[taglist[i][0]] += [i]
    target_df = ftrs['zw']
    feature_no = len(target_df.iloc[0])

    for act, nums in tag_class.items():
        for no in nums:
            feature[act].append(target_df.loc[no])
        feature[act] = np.array(feature[act])

    # corr across acts for different features
    # feature_across_acts = [Mat]: length is no of features
    # Mat is n*n (n=length of features)

    feature_across_acts = []
    for i in range(feature_no):
        Mat = []
        for act1, mat1 in feature.items():
            mat = []
            for act2, mat2 in feature.items():
                ft_1 = mat1[:, 0]
                ft_2 = mat2[:, 0]
                corrf = corrolate(ft_1, ft_2, corr=corr)
                mat.append(corrf)
            Mat.append(mat)
        feature_across_acts.append(np.array(Mat))

    feature_across_feature = []
    # mat is the sliding window series of features under specific act
    for act, mat in feature.items():
        M = []
        for i in range(feature_no):
            m = []
            for j in range(feature_no):
                ft_1 = mat[:, i]
                ft_2 = mat[:, j]
                corrf = corrolate(ft_1, ft_2, corr=corr)
                m.append(corrf)
            M.append(m)
        feature_across_feature.append(np.array(M))

    # print(feature_across_feature[0])
    n = [i for i in range(feature_no)]

    # plot the heatmap of any individual feature
    if cross == 'feature':
        heat = feature_across_feature[0]
    else:
        heat = feature_across_acts[0]
    label = n_class if (heat is feature_across_acts[0]) else n
    sns.set_theme(style="white")
    mask = np.zeros_like(heat)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(heat, annot=True, xticklabels=label, yticklabels=label, mask=mask)


if __name__ == '__main__':

    ft1 = pd.read_pickle(r'multiclass_new_tag\feature_child1_w5_norm.pickle')
    tag1 = pd.read_pickle(r'multiclass_new_tag\tag_child1_w5_norm.pickle')
    n_class = ['others', 'rope_skipping', 'sit_up1']

    # ft1 = pd.read_pickle(r'multiclass_new_tag\feature1_w10.pickle')
    # tag1 = pd.read_pickle(r'multiclass_new_tag\tag1_w10.pickle')
    # n_class = ['jog', 'others', 'rope_skipping', 'sit_up1', 'sit_up2', 'walking']

    # for mono dataset only, do not combine dataset
    feature_corr(ft1, tag1, n_class, cross='acts', corr='spearman')
    plt.figure()
    feature_corr(ft1, tag1, n_class, cross='acts', corr='kendall')
    plt.figure()
    feature_corr(ft1, tag1, n_class, cross='acts', corr='dtw')
    plt.show()

