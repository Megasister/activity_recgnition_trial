import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from copy import deepcopy

ftlist = pd.read_pickle(r'multiclass_new_tag\feature_child1_w[0.5, 1, 3, 5, 8, 10]_None.pickle')
taglist = pd.read_pickle(r'multiclass_new_tag\tag_child1_w[0.5, 1, 3, 5, 8, 10]_None.pickle')
# ft1 = pd.read_pickle(r'multiclass_new_tag\feature_child1_w10_test.pickle')
# tag1 = pd.read_pickle(r'multiclass_new_tag\tag_child1_w10_test.pickle')

sig_coeff = 0.05
axises = ['xa', 'ya', 'za', 'xw', 'yw', 'zw']
n_class = ['jog', 'others', 'rope_skipping', 'sit_up1', 'walking']
window_sizes = [0.5, 1, 3, 5, 8, 10]
feature = ['t_mean', 't_var', 't_std', 't_max', 't_min', 'f_dc', 'f_std()', 'f_skew', 'f_kurt', 'f_entropy', 'f_energy']


def get_feature_cube(Ord, ftlist, taglist, n_class):
    '''
    输入
        i : 窗口序号
        n_class 代表五种运动
    输出：
        mat = ft_cube[axis][act]
        ft_cube由两个dictionary嵌套组成，mat为二维数组
        mat: #features * #samples

    notice: Ord 单纯为上一步处理之下留下的行号，对数据维度不影响，
                但是必须有才能取得仅有一列（行）的pandas数据
    '''
    ftlist = ftlist.T[Ord].apply(pd.Series).T
    # print(ftlist)
    ftrs = {'xa':[], 'ya':[], 'za':[], 'xw':[], 'yw':[], 'zw':[]}
    for i, k in enumerate(ftrs.keys()):
        ftrs[k] = ftlist.iloc[i]

    tag_class, feature = {a:[] for a in n_class}, {a:[] for a in n_class}

    # cluster by activity
    for k, v in taglist.items():
        tag_class[v[Ord]] += [k]

    ft_cube = {}
    for axis in ftrs.keys():
        feature = {a: [] for a in n_class}
        target_df = ftrs[axis]
        for act, nums in tag_class.items():
            for no in nums:
                feature[act].append(target_df.loc[no])
            feature[act] = np.array(feature[act])

        ft_cube[axis] = feature
    feature_no = ft_cube[axises[0]][n_class[0]].shape[1]    # suggest shape of the ft_cube
    return ft_cube, feature_no


def norm_distribution_test(_series):
    # 连续属性是否符合正态分布
    # 样本大于5000：Kolmogorov-Smirnov test
    # 样本小于5000：shapiro-wilk
    sample_size = len(_series)
    # ks_stat, p_value = stats.shapiro(_series)
    if sample_size > 5000:
        ks_stat, p_value = stats.kstest(_series, 'norm')
    else:
        s_stat, p_value = stats.shapiro(_series)
    # p_value = round(p_value, 3)
    return p_value


def get_MAT(axis, act):
    MAT = []
    for i in range(len(window_sizes)):
        df_ft = pd.DataFrame(ftlist).loc[i].dropna(how='all')
        df_tag = pd.DataFrame(taglist).loc[i].dropna(how='all')
        df_ft = pd.DataFrame(df_ft).T  # 改变了数据结构，使得 get_cube时候必须要i序号取列，并非窗口循环用途，整体维度没有增加
        df_tag = pd.DataFrame(df_tag).T
        ft_cube, ft_no = get_feature_cube(i, df_ft, df_tag, n_class)
        mat = ft_cube[axis][act]
        Mat = []
        for feature_ord in range(ft_no):
            Mat.append(mat[:, feature_ord])
        MAT.append(Mat)
    return MAT


def get_combine_axis_feature_cube(window_ord):
    '''
    output: # activity * [axis * #features * #samples]
    '''
    # 对于每一种运动
    res = {}
    for act in n_class:
        All_feature = pd.DataFrame()
        for axis in axises:
            MAT = pd.DataFrame(get_MAT(axis, act)[window_ord])
            All_feature = pd.concat([All_feature, MAT], axis=0)
        res[act] = np.array(All_feature)
    return res


def across_axis_norm_analysis():
    # 对于每一种运动
    for act in n_class:
        plt.figure()
        # 对于每一个轴
        for axis in axises:
            MAT = get_MAT(axis, act)
            sig_val = []
            for i in range(len(window_sizes)):
                shapiro_val = [norm_distribution_test(mat) for mat in MAT[i]]
                print(MAT[i])
                sig_val.append(sum(i > sig_coeff for i in shapiro_val))

                mat = pd.DataFrame(MAT[i])

            # plt.figure()
            # plt.plot(shapiro_val)
            # plt.xticks([i for i in range(len(shapiro_val))], feature)
            # plt.title('shapiro test p value for feature from 1 to 11 of window_size {}'.format(window_sizes[i]))
            print(sig_val)
            # plt.figure()
            plt.plot(window_sizes, sig_val, label=axis)
            plt.scatter(window_sizes, sig_val)
            plt.xticks(window_sizes)
            plt.xlabel('window sizes (s)')
            plt.ylabel('number of sigfinicant features (>{})'.format(sig_coeff))

        plt.legend()
        plt.title('number of significant features across all window sizes\n'
                  'for {}'.format(act))
        plt.show()
        # plt.savefig(r'C:\Users\20804370\Desktop\试验结果\norm test显著特征数量vs窗口长度'
        #             '\{}_across_axis.png'.format(act), bbox_inches='tight')


if __name__ == '__main__':
    # across_axis_norm_analysis()
    window_ord = 3
    act_cube_dic = get_combine_axis_feature_cube(window_ord)
    assert (window_sizes[window_ord] == 5)

    # color = ['r', 'g', 'b', '']
    # color_dic = {}

    data_list = []
    for act1 in n_class:
        # other_class = deepcopy(n_class)
        # del
        for act2 in n_class:
            cube1 = act_cube_dic[act1]
            cube2 = act_cube_dic[act2]
            for i in range(len(cube1[:, 0])):
                vec1 = cube1[i, :]
                vec2 = cube2[i, :]
                _, p_value = stats.ks_2samp(vec1, vec2)
                point = (i, p_value)
                print(point)
                plt.scatter(i, p_value)

    plt.plot([sig_coeff for _ in range(66)], linestyle='--')
    plt.yscale('log')
    plt.show()









