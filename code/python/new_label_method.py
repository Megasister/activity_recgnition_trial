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
from scipy.stats import pearsonr, spearmanr
from scipy import signal

n_class = ['jog', 'others', 'rope_skipping', 'sit_up1', 'sit_up2', 'walking']
# n_class = ['others', 'rope_skipping', 'sit_up1']


def get_data(link):
    file = open(
        # second data series
        link,
        encoding="UTF-8",
        errors='ignore'
    )
    line = file.readline()
    GYRO, ACCEL = [], []
    while line:
        # print(line)
        if 'QSensorTest:' in line.split():
            if "GYRO" in line.split():
                GYRO.append(line.split()[:2] + line.split()[6:])
                # print(line.split()[:2] + line.split()[6:])
            if 'ACCEL' in line.split():
                ACCEL.append(line.split()[:2] + line.split()[6:])
                # print(line.split())

        line = file.readline()
    file.close()
    return GYRO, ACCEL


# 获得不同运动时间段对应的开始和结束时间戳
def get_timestamp_step_before_container(container, start_time):
    start = 0
    for i in range(len(container)):
        if pd.to_datetime(container[i][1]) > start_time:
            start = i
            break
    return start


# align two container at beginning
def align_init(g_start_time, a_start_time, accel, gyro):
    ACC, GY = [], []
    if g_start_time > a_start_time:
        for i in range(len(accel)):
            if pd.to_datetime(accel[i][1]) > g_start_time:
                ACC = accel[i:]
                break
        return ACC, gyro
    else:
        for k in range(len(gyro)):
            if pd.to_datetime(gyro[k][1]) > a_start_time:
                GY = gyro[k:]
                break
        return accel, GY


# align GY or ACC when one is upfront too far
# arg1: the one upfront at designated time slot
# arg2: one lag back
# arg3: designated time slot
def align_by_time(accel, gyro, interval_time, frequency, delay_max_time):
    ACC, GY, cut_point = [], [], 0
    interval_start = get_timestamp_step_before_container(accel, interval_time)

    if pd.to_datetime(accel[interval_start][1]) < pd.to_datetime(gyro[interval_start][1]):
        print('wrong order')
    for i in range(interval_start, interval_start + frequency * delay_max_time):
        if pd.to_datetime(accel[interval_start][1]) < pd.to_datetime(gyro[i][1]):
            cut_point = i
            break
    # print(interval_start)
    return gyro[:interval_start] + gyro[cut_point:]


def time_validate(container_a, container_g):
    tolerance = pd.Timedelta('00:00:05')
    for i in range(len(container_a)):
        try:
            diff = pd.to_datetime(container_g[i][1]) - pd.to_datetime(container_a[i][1])
            if diff > tolerance:
                print(i)
                print(container_a[i])
                print(container_g[i])
                print(diff)
        except:
            continue


# 获得合成矢量
def sumvector(*args):
    return math.sqrt(sum([pow(arg, 2) for arg in args]))


def get_timestamp_step(container, start_time, end_time):
    start, end = 0, 0
    for i in range(len(container)):
        if container[i][0] > start_time:
            start = i
            break
    for k in range(start, len(container)):
        if container[k][0] > end_time:
            end = k
            break
    return start, end


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


# get data on each axis
def get_coor(container_A1, container_G1, reg=None):
    xa, ya, za = np.array(container_A1)[:, 1], np.array(container_A1)[:, 2], np.array(container_A1)[:, 3]
    xw, yw, zw = np.array(container_G1)[:, 1], np.array(container_G1)[:, 2], np.array(container_G1)[:, 3]
    if reg == 'stdr':
        xa, ya, za = standardization(xa), standardization(ya), standardization(za)
        xw, yw, zw = standardization(xw), standardization(yw), standardization(zw)
    elif reg == 'norm':
        xa, ya, za = normalization(xa), normalization(ya), normalization(za)
        xw, yw, zw = normalization(xw), normalization(yw), normalization(zw)
    return xa, ya, za, xw, yw, zw


# 根据时间窗口取得19个时域频域特征
def get_aw_feature(walk_start_stamp, walk_end_stamp, container_A1, container_G1):
    xa, ya, za, xw, yw, zw = get_coor(container_A1, container_G1)
    xa, ya, za = xa[walk_start_stamp:walk_end_stamp], ya[walk_start_stamp:walk_end_stamp], za[walk_start_stamp:walk_end_stamp]
    xw, yw, zw = xw[walk_start_stamp:walk_end_stamp], yw[walk_start_stamp:walk_end_stamp], zw[walk_start_stamp:walk_end_stamp]

    # 六个加速度的统计特征 （窗口为500，步长为500）
    feature_xa = feature_core.sequence_feature(xa, 0, 0)
    feature_ya = feature_core.sequence_feature(ya, 0, 0)
    feature_za = feature_core.sequence_feature(za, 0, 0)
    feature_xw = feature_core.sequence_feature(xw, 0, 0)
    feature_yw = feature_core.sequence_feature(yw, 0, 0)
    feature_zw = feature_core.sequence_feature(zw, 0, 0)

    # return [feature_xa]
    return [feature_xa, feature_ya, feature_za, feature_xw, feature_yw, feature_zw]


# 取得四元数
def q_iteration(q0, q1, q2, q3, wx, wy, wz, delta):
    q0 = q0 + (-wx * q1 - wy * q2 - wz * q3) * delta / 2
    q1 = q1 + ( wx * q0 - wy * q3 + wz * q2) * delta / 2
    q2 = q2 + ( wx * q3 + wy * q0 - wz * q1) * delta / 2
    q3 = q3 + (-wx * q2 + wy * q1 - wz * q0) * delta / 2
    # 归一化
    s = sumvector(q0, q1, q2, q3)
    return q0 / s, q1 / s, q2 / s, q3 / s


# 根据四元数算出姿态角每步的变化量
def get_quadranion(container_G):
    q0, q1, q2, q3 = 1, 0, 0, 0
    Q = []
    for g in container_G:
        gx, gy, gz = g[1], g[2], g[3]
        delta = 0.02
        q0, q1, q2, q3 = q_iteration(q0, q1, q2, q3, gx, gy, gz, delta)
        Q.append([q0, q1, q2, q3])

    delta_angle = []
    for q in Q:
        q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
        del_theta = -math.asin(2 * (q1 * q2 - q0 * q3))
        del_fai = math.atan(2 * (q1 * q3 - q0 * q1) / (q3 ** 2 + q2 ** 2 - q1 ** 2 + q0 ** 2))
        del_psi = math.atan(2 * (q1 * q2 - q0 * q3) / (q1 ** 2 + q0 ** 2 - q3 ** 2 - q2 ** 2))
        delta_angle.append([del_theta, del_fai, del_psi])

    return delta_angle


# label the slidding window, return tag
def label_target_activity(stamp, window_size, overlap_tolerance, head_tail_extend):
    # initialize overall start and end stamp with head_tail_extend
    start, end, taglist = [], [], []
    for extend in head_tail_extend:
        for i, (_, act) in enumerate(stamp.items()):
            if i == 0:
                start.append(0 if act[0] - extend < 0 else act[0] - extend)
            if i == len(stamp)-1:
                end.append(act[1] + extend)
    # tag corresponding activity for sliding window
    # that has an overlap with activity larger than overlap_tolerance = frequency * react_time
    # and other position 'others'
    for j in range(len(start)):
        tags = {}
        for i in range(start[j], end[j], window_size[j]//2):
            tags[i], s, e = 'others', i, i + window_size[j]
            for name, act in stamp.items():
                if (e - overlap_tolerance[j] < act[0]) or (s + overlap_tolerance[j] > act[1]):
                    continue
                else:
                    # tags[i] = 'rope_skipping'
                    tags[i] = name
        taglist.append(tags)
    return taglist


def preprocess(link):
    G1, A1 = get_data(link)
    # Align the timestamp as close as possible uniformly conforming to the latter one
    G1_start_time = pd.to_datetime(G1[0][1])
    A1_start_time = pd.to_datetime(A1[0][1])
    # print(GYRO_start_time)
    # print(ACCEL_start_time)
    ACC1, GY1 = align_init(G1_start_time, A1_start_time, A1, G1)
    GY1 = align_by_time(ACC1, GY1, pd.to_datetime('07:43:50'), frequency=50, delay_max_time=1000)
    ACC1 = align_by_time(GY1, ACC1, pd.to_datetime('07:59:22.108'), frequency=50, delay_max_time=1000)

    # time_validate(ACC1, GY1)
    # contains time and coord
    container_A1, container_G1 = [], []
    for category in [ACC1, GY1]:
        for data in category:
            nums = data[3].split(',')
            x = float(nums[0][13:])
            y = float(nums[1][4:])
            z = float(nums[2][4:])
            if category == ACC1:
                container_A1.append([pd.to_datetime(data[1]), x, y, z])
            if category == GY1:
                container_G1.append([pd.to_datetime(data[1]), x, y, z])

    return container_A1, container_G1


# Print confusion matrix
def plot_cf_mat(clf, X_test, Y_test):
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    titles_options = [("Confusion matrix, without normalization", None),
                      ("Normalized confusion matrix", 'true')]
    # confusion matrix
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(clf, X_test, Y_test,
                                     cmap='Blues',
                                     normalize=normalize)
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)


def plot_roc(Y_test, Y_predict, Y_prob, n_class, window_size):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    Y_test = label_binarize(Y_test, classes=n_class)
    Y_predict = label_binarize(Y_predict, classes=n_class)

    # print(Y_test)
    # print(Y_prob[:, 0])
    # print(roc_auc_score(Y_test, Y_prob, average='micro'))
    plt.figure()
    for j in range(len(n_class) if len(n_class) > 2 else len(n_class)-1):
        if len(n_class) > 2:
            y_true = Y_test[:, j]
            y_prob = Y_prob[:, j]
        else:
            y_true = Y_test[:, 1]
            y_prob = Y_prob[:, 1]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        print(roc_auc_score(y_true, y_prob))

        # plot the roc curve for the model
        plt.plot(fpr, tpr, linestyle='--', label='classes of {}: auc = {}'.format(n_class[j], roc_auc))
        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # show the legend
        plt.legend()
    plt.title('ROC curve for window_size of {}'.format(window_size))

    f, t, _ = roc_curve(Y_test.ravel(), Y_prob.ravel())
    roc_auc = auc(f, t)
    plt.plot(f, t, linestyle='--', label='micro roc: auc = {}'.format(roc_auc))
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    plt.title('ROC curve for window_size of {}'.format(window_size))
    plt.show()


def convert_multiclass_binary(df_tags, n_class, i):
    for k in range(len(df_tags)):
        if df_tags.iloc[k][0] != n_class[i]:
            df_tags.iloc[k][0] = 'others'

    return df_tags


def load_data():
    container_A1, container_G1 = preprocess(parameter.link1)
    print('write over')
    for x in container_A1:
        print(x)

    # # 取得四元数
    # delta_angle = get_quadranion(container_G1)

    stamp = {}
    # # for link_sk
    # s1, e1, s2, e2, s3, e3, s4, e4 = 1100, 1600, 2000, 3300, 3880, 4050, 4500, 5440
    # stamp['rope_skipping'] = [s1, e1]
    # stamp['rope_skipping1'] = [s2, e2]
    # stamp['rope_skipping2'] = [s3, e3]
    # stamp['rope_skipping3'] = [s4, e4]

    # for link1, conservative policy which confines the range of each activity clearly
    # allow sliding window to swipe the entire range
    # if sliding window has an overlapping range with specified ranges over 2s, tag the window as positive
    s1, e1, s2, e2, s3, e3, s4, e4, s5, e5 = 1550, 3520, 6900, 9350, 11480, 17580, 20800, 25260, 28670, 34740
    stamp['sit_up1'] = [s1, e1]
    stamp['sit_up2'] = [s2, e2]
    stamp['jog'] = [s3, e3]
    stamp['rope_skipping'] = [s4, e4]
    stamp['walking'] = [s5, e5]

    print(container_A1)

    # # for link2
    # s1, e1, s2, e2, s3, e3, s4, e4 = 37900, 40150, 42950, 45050, 52400, 58450, 65040, 66350
    # stamp['sit_up1'] = [s1, e1]
    # stamp['sit_up2'] = [s2, e2]
    # stamp['jog'] = [s3, e3]
    # stamp['rope_skipping'] = [s4, e4]

    # 调试绘图
    # sos = signal.butter(3, 1, 'lp', fs=52, output='sos')
    # accdata = signal.sosfilt(sos, [g[-1] for g in container_G1])
    # plt.plot([sumvector(a[1], a[2], a[3]) for a in container_A1], label='sum acceleration')
    # plt.plot([g[2] for g in container_G1], label='z angular velocity')
    # plt.plot(accdata, label='high pass filter wave')
    # plt.legend()
    # plt.plot([a[1] for a in container_G1], label='wx')
    # plt.plot([a[2] for a in container_G1], label='wy')
    # plt.plot([a[3] for a in container_G1], label='wz')
    # plt.plot([d[0] for d in delta_angle], label='theta')
    # plt.plot([d[1] for d in delta_angle], label='fai')
    # plt.plot([d[2] for d in delta_angle], label='psi')
    plt.show()

    # window_time = [0.5,1,3,5,8,10]  # s time of a window frame
    window_time = [10]
    print('window time is:', window_time)
    overlap_rate = 0.2
    react_time = np.multiply(overlap_rate, window_time)  # s time for algrm to get enough data (for window analysis)
    print('react time is:', react_time)
    frequency = 52  # Hz designated frequency of sampling
    window_size = np.rint(np.multiply(window_time, frequency)).astype(int)  # number of samples in a window frame
    overlap_tolerance = np.rint(np.multiply(react_time, frequency)).astype(int)  # minimum number of samples for overlap the act would be considered on
    extend_coefficient = 1.5
    head_tail_extend = np.multiply(window_size, extend_coefficient).astype(int)  # number of samples used to over samling before and after the series to acquire more generalized data
    # assert (window_time > react_time)  # window frame is always longer than the reaction period

    # tag dict
    taglist = label_target_activity(stamp, window_size, overlap_tolerance, head_tail_extend)
    # feature dict
    ftlist = []
    for i in range(len(taglist)):
        tags, size, ft = taglist[i], window_size[i], {}
        print('This is window time of {}s'.format(window_time[i]))
        for k, _ in tags.items():
            print(k)
            ft[k] = get_aw_feature(k, k+size, container_A1, container_G1)
        ftlist.append(ft)

    # print(ftlist)
    print('feature num is:', len(next(iter(ftlist[0].values()))[0]))
    print('total dim is:', len(next(iter(ftlist[0].values()))))
    # save the feature and tags in pickel file
    # pd.DataFrame(ftlist).to_pickle(r'.\multiclass_new_tag\feature2_w5.pickle')
    # pd.DataFrame(taglist).to_pickle(r'.\multiclass_new_tag\tag2_w5.pickle')
    # print('preprocess over')


def feature_eng(ftlist, taglist):
    ftlist = ftlist.T[0].apply(pd.Series).T
    ftrs = {'xa':[], 'ya':[], 'za':[], 'xw':[], 'yw':[], 'zw':[]}
    for i, k in enumerate(ftrs.keys()):
        print(i, k)
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
            print(mat1)
            for act2, mat2 in feature.items():
                print(mat1)
                ft_1 = mat1[:, 0]
                ft_2 = mat2[:, 0]
                # get the front overlapping slice
                min_len = min(len(ft_1), len(ft_2))
                # corr, _ = pearsonr(ft_1[:min_len], ft_2[:min_len])
                corr = spearmanr(ft_1[:min_len], ft_2[:min_len])
                mat.append(corr)
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
                # corr1, _ = pearsonr(ft_1, ft_2)
                corr, _ = spearmanr(ft_1, ft_2)
                m.append(corr)
            M.append(m)
            # print(np.array(M).shape)
        feature_across_feature.append(np.array(M))

    # print(feature_across_feature[0])
    n = [i for i in range(feature_no)]

    # plot the heatmap of any individual feature
    heat = feature_across_feature[0]
    label = n_class if (heat is feature_across_acts[0]) else n
    sns.set_theme(style="white")
    mask = np.zeros_like(heat)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(heat, annot=True, xticklabels=label, yticklabels=label, mask=mask)


def get_train_test_data(ftlist, taglist, i, n_class, test_size, random_state, shuffle_state):
    try:
        df_ft = shuffle(pd.DataFrame(ftlist[i]).T, random_state=shuffle_state).T
        df_tags = shuffle(pd.DataFrame(taglist[i], index=[0]).T, random_state=shuffle_state)
    except:
        df_ft = shuffle(pd.DataFrame(ftlist).loc[i].dropna(how='all'), random_state=shuffle_state)
        df_tags = shuffle(pd.DataFrame(taglist).loc[i].dropna(how='all'), random_state=shuffle_state)

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


def process_data():
    # take data and process
    # ft1 = pd.read_pickle(r'multiclass_new_tag\feature_child1_w5.pickle')
    # tag1 = pd.read_pickle(r'multiclass_new_tag\tag_child1_w5.pickle')

    ft1 = pd.read_pickle(r'multiclass_new_tag\feature1_w10.pickle')
    tag1 = pd.read_pickle(r'multiclass_new_tag\tag1_w10.pickle')
    # ftlist = [ftlist]
    # taglist = [taglist]

    # get correlation heatmap of features vs acts and features vs features
    feature_eng(ft1, tag1)
    plt.show()

    # ft1 = pd.read_pickle(r'multiclass_new_tag\feature_sk_w5.pickle')
    # tags1 = pd.read_pickle(r'multiclass_new_tag\tag_sk_w5.pickle')
    ft2 = pd.read_pickle(r'multiclass_new_tag\feature_child2_w5.pickle')
    tag2 = pd.read_pickle(r'multiclass_new_tag\tag_child2_w5.pickle')
    ft3 = pd.read_pickle(r'multiclass_new_tag\feature_child3_w5.pickle')
    tag3 = pd.read_pickle(r'multiclass_new_tag\tag_child3_w5.pickle')
    ft4 = pd.read_pickle(r'multiclass_new_tag\feature1_w5.pickle')
    tag4 = pd.read_pickle(r'multiclass_new_tag\tag1_w5.pickle')
    ft5 = pd.read_pickle(r'multiclass_new_tag\feature2_w5.pickle')
    tag5 = pd.read_pickle(r'multiclass_new_tag\tag2_w5.pickle')
    ftlist = pd.concat([ft1, ft2, ft3, ft4, ft5], axis=1, sort=False)
    taglist = pd.concat([tag1, tag2, tag3, tag4, tag5], axis=1, sort=False)

    # ORDER MATTERS
    window_sizes = [0.5,1,3,5,8,10]

    for i in range(len(ftlist)):
        print('window size is:', window_sizes[i])
        X_train, X_test, Y_train, Y_test = get_train_test_data(ftlist, taglist, i, n_class, test_size=0.3,
                                                               random_state=7, shuffle_state=23)
        # clf = svm.SVC(kernel='rbf', gamma=1.0)
        # clf = OneVsRestClassifier(GaussianNB())
        # clf = MultinomialNB()
        # clf = GaussianNB()
        # clf = BernoulliNB()
        clf = RandomForestClassifier()
        # Train the model using the training sets
        clf.fit(X_train, Y_train)
        Y_predict = clf.predict(X_test)
        Y_prob = clf.predict_proba(X_test)

        # Y_test = label_binarize(Y_test, classes=n_class)
        # Y_predict = label_binarize(Y_predict, classes=n_class)
        # cr_sc = cross_val_score(clf, X_test , Y_test, cv=5)
        # print(cr_sc)

        plot_cf_mat(clf, X_test, Y_test)
        report = classification_report(Y_test, Y_predict)
        print(report)
        if len(n_class) > 2:
            plot_roc(Y_test, Y_predict, Y_prob, n_class, window_sizes[i])
        else:
            plot_roc_curve(clf, X_test, Y_test)
        plt.show()


if __name__ == '__main__':
    # load_data()
    process_data()

