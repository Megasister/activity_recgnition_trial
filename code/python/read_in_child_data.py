import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import scipy.io
from feature_extraction.tsfeature import feature_core, feature_fft, feature_time
import parameter
from copy import deepcopy


def get_container(link):
    lines = pd.read_csv(link)
    # print(lines)

    lines.columns = ['ord', 'time', 'ax', 'ay', 'az', 'wx', 'wy', 'wz', 'tile']
    lines.drop(['ord', 'time', 'tile'], axis=1, inplace=True)

    container_A1, container_G1 = [], []
    a_axis = ['ax', 'ay', 'az']
    g_axis = ['wx', 'wy', 'wz']
    for axis in a_axis:
        container_A1.append(pd.Series(lines[axis], name=axis))
    for axis in g_axis:
        container_G1.append(pd.Series(lines[axis], name=axis))

    container_A1 = np.array(container_A1)
    container_A1 = container_A1.T
    container_G1 = np.array(container_G1)
    container_G1 = container_G1.T

    return container_A1, container_G1


# label the slidding window, return tag, start by a boundary closer to target activity, include more 'others' sample
def label_target_activity(stamp, window_size, overlap_tolerance, head_tail_extend):
    # initialize overall start and end stamp with head_tail_extend
    start, end, taglist = [], [], []
    for extend in head_tail_extend:
        s_bucket, e_bucket = [], []
        for _, (_, act) in enumerate(stamp.items()):
            for i, tup in enumerate(act):
                if i == 0:
                    s_bucket.append(0 if tup[0] - extend < 0 else tup[0] - extend)
                if i == len(act)-1:
                    e_bucket.append(tup[1] + extend)
        start.append(s_bucket)
        end.append(e_bucket)
    print(start)
    print(end)
    # tag corresponding activity for sliding window
    # that has an overlap with activity larger than overlap_tolerance = frequency * react_time
    # and other position 'others'

    # k control the window size
    for k in range(len(start)):
        # j control activity_chunk boundary
        tags = {}
        for j in range(len(start[k])):
            # i control slide window position
            for i in range(start[k][j], end[k][j], window_size[k]//2):
                tags[i], s, e = 'others', i, i + window_size[k]
                for name, act in stamp.items():
                    for tup in act:
                        if (e - overlap_tolerance[k] < tup[0]) or (s + overlap_tolerance[k] > tup[1]):
                            continue
                        else:
                            if name.startswith('r'):
                                tags[i] = 'rope_skipping'
                            if name.startswith('w'):
                                tags[i] = 'walking'
                            if name.startswith('s'):
                                tags[i] = 'sit_up1'
                            if name.startswith('j'):
                                tags[i] = 'jog'
                            break
        print(tags)
        taglist.append(tags)
    return taglist


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


# get data on each axis
def get_coor(container_A1, container_G1, reg=None):
    xa, ya, za = np.array(container_A1)[:, 0], np.array(container_A1)[:, 1], np.array(container_A1)[:, 2]
    xw, yw, zw = np.array(container_G1)[:, 0], np.array(container_G1)[:, 1], np.array(container_G1)[:, 2]
    if reg == 'stdr':
        xa, ya, za = standardization(xa), standardization(ya), standardization(za)
        xw, yw, zw = standardization(xw), standardization(yw), standardization(zw)
    elif reg == 'norm':
        xa, ya, za = normalization(xa), normalization(ya), normalization(za)
        xw, yw, zw = normalization(xw), normalization(yw), normalization(zw)
    return xa, ya, za, xw, yw, zw


# 根据时间窗口取得19个时域频域特征
def get_aw_feature(walk_start_stamp, walk_end_stamp, container_A1, container_G1, local_norm=None, flatten=False):
    xa, ya, za, xw, yw, zw = get_coor(container_A1, container_G1, local_norm)
    xa, ya, za = xa[walk_start_stamp:walk_end_stamp], ya[walk_start_stamp:walk_end_stamp], za[walk_start_stamp:walk_end_stamp]
    xw, yw, zw = xw[walk_start_stamp:walk_end_stamp], yw[walk_start_stamp:walk_end_stamp], zw[walk_start_stamp:walk_end_stamp]

    # 六个加速度的统计特征 （窗口为500，步长为500）
    for i, (pxa, pya, pza, pxw, pyw, pzw) in enumerate(zip(xa, ya, za, xw, yw, zw)):
        print(pxa, pya, pza, pxw, pyw, pzw)

    feature_xa = feature_core.sequence_feature(xa, 0, 0)
    feature_ya = feature_core.sequence_feature(ya, 0, 0)
    feature_za = feature_core.sequence_feature(za, 0, 0)
    feature_xw = feature_core.sequence_feature(xw, 0, 0)
    feature_yw = feature_core.sequence_feature(yw, 0, 0)
    feature_zw = feature_core.sequence_feature(zw, 0, 0)

    # print("feature:"),
    # for i in [feature_xa.tolist() + feature_ya.tolist() + feature_za.tolist()
    #         + feature_xw.tolist() + feature_yw.tolist() + feature_zw.tolist()][0]:
    #     print(i)
    # print("*************************************")

    if not flatten:
        return [feature_xa, feature_ya, feature_za, feature_xw, feature_yw, feature_zw]
    return [feature_xa.tolist() + feature_ya.tolist() + feature_za.tolist()
            + feature_xw.tolist() + feature_yw.tolist() + feature_zw.tolist()]


def norm_container(container_A1, container_G1, s, e, reg):
    if s <= 0: s = 1
    container_A, container_G = deepcopy(container_A1), deepcopy(container_G1)
    xa, ya, za = container_A1[s-1:e, 0], container_A1[s-1:e, 1], container_A1[s-1:e, 2]
    xw, yw, zw = container_G1[s-1:e, 0], container_G1[s-1:e, 1], container_G1[s-1:e, 2]
    if reg=='stdr':
        xa, ya, za = standardization(xa), standardization(ya), standardization(za)
        xw, yw, zw = standardization(xw), standardization(yw), standardization(zw)
    elif reg=='norm':
        xa, ya, za = normalization(xa), normalization(ya), normalization(za)
        xw, yw, zw = normalization(xw), normalization(yw), normalization(zw)
    if reg:
        container_A[s-1:e, 0], container_A[s-1:e, 1], container_A[s-1:e, 2] = xa, ya, za
        container_G[s-1:e, 0], container_G[s-1:e, 1], container_G[s-1:e, 2] = xw, yw, zw
    # plt.plot(container_A1[s-1:e, 0])
    # plt.figure()
    # plt.plot(container_A[s-1:e, 0])
    # plt.show()
    return container_A, container_G


def get_norm_container(container_A1, container_G1, stamp, reg):
    container_A, container_G = deepcopy(container_A1), deepcopy(container_G1)
    for list_tuple in stamp.values():
        print(list_tuple)
        s, e = list_tuple[0][0], list_tuple[-1][-1]
        print(s, e)
        container_A, container_G = norm_container(container_A, container_G, s, e, reg)
    return container_A, container_G


def process():
    norm_slide_window = None
    global_norm = 'None'
    # window_time = [0.5, 1, 3, 5, 8, 10]  # s time of a window frame
    window_time = [4.92]
    kid_link = parameter.kid_link5
    kid_no = 5
    print(kid_no, norm_slide_window)

    stamp = {}

    # # kid_link5
    s1, e1, s2, e2= 609, 37730, 52600, 78370
    stamp['walking'] = [(s1, e1)]
    stamp['jog'] = [(s2, e2)]

    container_A1, container_G1 = get_container(kid_link)

    # mu, var = get_stdr_para(container_A1, container_G1, start, end)

    # 调试绘图
    # sos = signal.butter(3, 1, 'lp', fs=52, output='sos')
    # accdata = signal.sosfilt(sos, [g[-1] for g in container_G1])
    # plt.plot([sumvector(a[1], a[2], a[3]) for a in container_A1], label='sum acceleration')
    # plt.plot([g[2] for g in container_G1[s1-ext:e3+ext]], label='z angular velocity')
    # plt.plot(accdata, label='high pass filter wave')
    # plt.legend()
    # plt.plot([a[1] for a in container_G1], label='wx')
    # plt.plot([a[2] for a in container_G1], label='wy')
    # plt.plot([a[3] for a in container_G1], label='wz')
    # plt.plot([d[0] for d in delta_angle], label='theta')
    # plt.plot([d[1] for d in delta_angle], label='fai')
    # plt.plot([d[2] for d in delta_angle], label='psi')
    # plt.show()

    # window_time = [8]
    print('window time is:', window_time)
    overlap_rate = 0.2
    react_time = np.multiply(overlap_rate, window_time)  # s time for algrm to get enough data (for window analysis)
    print('react time is:', react_time)
    frequency = 52  # Hz designated frequency of sampling
    window_size = np.rint(np.multiply(window_time, frequency)).astype(int)  # number of samples in a window frame
    overlap_tolerance = np.rint(np.multiply(react_time, frequency)).astype(
        int)  # minimum number of samples for overlap the act would be considered on
    extend_coefficient = 1.5
    head_tail_extend = np.multiply(window_size, extend_coefficient).astype(
        int)  # number of samples used to over samling before and after the series to acquire more generalized data
    # assert (window_time > react_time)  # window frame is always longer than the reaction period

    # tag dict
    taglist = label_target_activity(stamp, window_size, overlap_tolerance, head_tail_extend)

    # observe the tag
    # for i in range(len(window_size)):
    #     c = 0
    #     for tag, value in taglist[i].items():
    #         alpha = 250
    #         if c > alpha:
    #             plt.figure()
    #             s, e = int(tag), int(tag) + window_size[i]
    #             _, _, _, _, _, wz = get_coor(container_A1, container_G1)
    #             plt.plot(wz[s:e])
    #             plt.title('{}, window size= {}'.format(value, window_size[i]))
    #         c += 1
    #         if c == alpha + 5:
    #             break
    #     plt.show()
    # i

    # feature dict
    ftlist = []
    for i in range(len(taglist)):
        tags, size, ft, extend = taglist[i], window_size[i], {}, head_tail_extend[i]
        norm_container_A1, norm_container_G1 = get_norm_container(container_A1, container_G1, stamp,
                                                                  global_norm)
        print('This is window time of {}s'.format(window_time[i]))
        for k, _ in tags.items():
            print(k)
            # print('window size ord', i)
            ft[k] = get_aw_feature(k, k + size, norm_container_A1, norm_container_A1, flatten=False)
        ftlist.append(ft)

    # print(ftlist)
    print('feature num is:', len(next(iter(ftlist[0].values()))[0]))
    print('total dim is:', len(next(iter(ftlist[0].values()))))

    # save the feature and tags in pickel file
    pd.DataFrame(ftlist).to_pickle(r'.\multiclass_new_tag\feature_child{}_w{}_{}.pickle'.format(kid_no, window_time, global_norm))
    pd.DataFrame(taglist).to_pickle(r'.\multiclass_new_tag\tag_child{}_w{}_{}.pickle'.format(kid_no, window_time, global_norm))
    print('preprocess over')


if __name__ == '__main__':
    process()

    # test between c and language
    # fft_trans = np.abs(np.fft.fft(test))
    # freq_spectrum = fft_trans[1:int(np.floor(len(test) * 1.0 / 2)) + 1]
    # print(freq_spectrum)
    # print(feature_core.sequence_feature(test, 0, 0))