import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
import re
import math
from feature_extraction.tsfeature import feature_core, feature_fft, feature_time
from sklearn import svm
from sklearn.metrics import SCORERS, confusion_matrix, plot_confusion_matrix, roc_auc_score, roc_curve, plot_roc_curve
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from dtw import distance, dtw
from sklearn.naive_bayes import MultinomialNB, GaussianNB


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


# 根据时间窗口取得19个时域频域特征
def get_aw_feature(walk_start_stamp, walk_end_stamp, container_A, container_G):
    accsignal = container_A[walk_start_stamp:walk_end_stamp]
    gyrosignal = container_G[walk_start_stamp:walk_end_stamp]
    xa, ya, za = [a[1] for a in accsignal], [a[2] for a in accsignal], [a[3] for a in accsignal]
    xa, ya, za = np.asarray(xa), np.asarray(ya), np.asarray(za)
    xw, yw, zw = [a[1] for a in gyrosignal], [a[2] for a in gyrosignal], [a[3] for a in gyrosignal]
    xw, yw, zw = np.asarray(xw), np.asarray(yw), np.asarray(zw)

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
    q1 = q1 + (wx * q0 - wy * q3 + wz * q2) * delta / 2
    q2 = q2 + (wx * q3 + wy * q0 - wz * q1) * delta / 2
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
def label_target_activity(stamp, container, window_length, right_class):
    start_stamp, end_stamp = stamp[0], stamp[1]
    tag, sumforce = {}, np.asarray([sumvector(a[1], a[2], a[3]) for a in container[:window_length]])

    # need to be certain which parameter to choose for small altitude
    # hypothesis (to be validated): 75 for only large shape and 50 for medium shape
    # which may indicates last motion to collect rope after continuous skipping
    # need to be tuned for every shape
    vault_coeff = 75
    for i in range(start_stamp, end_stamp, window_length // 2):
        stampend = end_stamp if i + window_length > end_stamp else (i + window_length)
        if np.asarray(
                [sumvector(a[1], a[2], a[3]) for a in container[i:stampend]]).max() > vault_coeff * sumforce.std():
            tag[i] = right_class
        else:
            tag[i] = 'others'
    return tag


def preprocess():
    link1 = r"C:\Users\20804370\Desktop\202012081708-ancfthrveijdbckh\xtcdata\Logs_Collector\Folder_20180223\Log.main_sys_2018-02-23__00-01-46.log"
    link2 = r"C:\Users\20804370\Desktop\202012081708-ancfthrveijdbckh\xtcdata\Logs_Collector\Folder_20180223\Log.main_sys_2018-02-23__08-11-42.log"
    link_sk = r"C:\Users\20804370\Desktop\202012211553-ancfthrveijdbckh\xtcdata\Logs_Collector\Folder_20201221\Log.main_sys_2020-12-21__10-49-32.log"

    G1, A1 = get_data(link1)
    G2, A2 = get_data(link2)

    # Align the timestamp as close as possible uniformly conforming to the latter one
    G1_start_time = pd.to_datetime(G1[0][1])
    A1_start_time = pd.to_datetime(A1[0][1])
    G2_start_time = pd.to_datetime(G2[0][1])
    A2_start_time = pd.to_datetime(A2[0][1])
    # print(GYRO_start_time)
    # print(ACCEL_start_time)
    ACC1, GY1 = align_init(G1_start_time, A1_start_time, A1, G1)
    ACC2, GY2 = align_init(G2_start_time, A2_start_time, A2, G2)
    GY1 = align_by_time(ACC1, GY1, pd.to_datetime('07:43:50'), frequency=50, delay_max_time=1000)
    ACC1 = align_by_time(GY1, ACC1, pd.to_datetime('07:59:22.108'), frequency=50, delay_max_time=1000)

    # time_validate(ACC1, GY1)
    # contains time and coord
    container_A1, container_A2, container_G1, container_G2 = [], [], [], []
    for category in [ACC1, ACC2, GY1, GY2]:
        for data in category:
            nums = data[3].split(',')
            x = float(nums[0][13:])
            y = float(nums[1][4:])
            z = float(nums[2][4:])
            if category == ACC1:
                container_A1.append([pd.to_datetime(data[1]), x, y, z])
            if category == ACC2:
                container_A2.append([pd.to_datetime(data[1]), x, y, z])
            if category == GY1:
                container_G1.append([pd.to_datetime(data[1]), x, y, z])
            if category == GY2:
                container_G2.append([pd.to_datetime(data[1]), x, y, z])

    return container_A1, container_A2, container_G1, container_G2


if __name__ == '__main__':
    container_A1, container_A2, container_G1, container_G2 = preprocess()
    print('write over')

    # # 取得四元数
    # delta_angle = get_quadranion(container_G1)

    # 不同运动分段时间调整，第一段时间
    situp1_start_time, situp1_end_time = pd.to_datetime('07:40:00'), pd.to_datetime('07:41:35')
    situp2_start_time, situp2_end_time = pd.to_datetime('07:42:15'), pd.to_datetime('07:43:30')
    jog_start_time, jog_end_time = pd.to_datetime('07:44:00'), pd.to_datetime('07:49:30')
    rope_skipping_start_time, rope_skipping_end_time = pd.to_datetime('07:49:50'), pd.to_datetime('07:52:05')
    walk_start_time, walk_end_time = pd.to_datetime('07:53:10'), pd.to_datetime('07:55:15')

    # # # #  第二段时间
    # situp1_start_time, situp1_end_time = pd.to_datetime('07:58:50'), pd.to_datetime('08:00:40')
    # situp2_start_time, situp2_end_time = pd.to_datetime('08:01:05'), pd.to_datetime('08:02:40')
    # jog_start_time, jog_end_time = pd.to_datetime('08:03:50'), pd.to_datetime('08:07:10')
    # rope_skipping_start_time, rope_skipping_end_time = pd.to_datetime('08:07:50'), pd.to_datetime('08:10:10')
    # #
    # # 获得不同运动分段时间戳
    stamp = {}
    situp1_start_stamp, situp1_end_stamp = get_timestamp_step(container_G1, situp1_start_time, situp1_end_time)
    situp2_start_stamp, situp2_end_stamp = get_timestamp_step(container_G1, situp2_start_time, situp2_end_time)
    jog_start_stamp, jog_end_stamp = get_timestamp_step(container_G1, jog_start_time, jog_end_time)
    rope_skipping_start_stamp, rope_skipping_end_stamp = get_timestamp_step(container_G1, rope_skipping_start_time,
                                                                            rope_skipping_end_time)
    walk_start_stamp, walk_end_stamp = get_timestamp_step(container_G1, walk_start_time, walk_end_time)
    # # rope_skipping_start_stamp2, rope_skipping_end_stamp2 = get_timestamp_step(container_G1, rope_skipping_start_time2,)
    #
    stamp['situp1'] = [situp1_start_stamp, situp1_end_stamp]
    stamp['situp2'] = [situp2_start_stamp, situp2_end_stamp]
    stamp['jog'] = [jog_start_stamp, jog_end_stamp]
    stamp['rope_skipping'] = [rope_skipping_start_stamp, rope_skipping_end_stamp]
    stamp['walk'] = [walk_start_stamp, walk_end_stamp]
    #
    # # stamp = {}
    # # s1, e1, s2, e2, s3, e3, s4, e4 = 1100, 1600, 2000, 3300, 3880, 4050, 4500, 5440
    # # stamp['rope_skipping'] = [s1, e1]
    # # stamp['rope_skipping1'] = [s2, e2]
    # # stamp['rope_skipping2'] = [s3, e3]
    # # stamp['rope_skipping3'] = [s4, e4]

    '''
    # 输出mat文件
    # scipy.io.savemat('胸前仰卧起坐.mat', {'GYRO':container_G1[situp1_start_stamp:situp1_end_stamp], 'ACCEL':container_A1[situp1_start_stamp:situp1_end_stamp]})
    # scipy.io.savemat('抱头仰卧起坐.mat', {'GYRO':container_G1[situp2_start_stamp:situp2_end_stamp], 'ACCEL':container_A1[situp2_start_stamp:situp2_end_stamp]})
    # scipy.io.savemat('跑步.mat', {'GYRO':container_G1[jog_start_stamp:jog_end_stamp], 'ACCEL':container_A1[jog_start_stamp:jog_end_stamp]})
    # scipy.io.savemat('跳绳.mat', {'GYRO':container_G1[rope_skipping_start_stamp:rope_skipping_end_stamp], 'ACCEL':container_A1[rope_skipping_start_stamp:rope_skipping_end_stamp]})
    # scipy.io.savemat('走路.mat', {'GYRO':container_G1[walk_start_stamp:walk_end_stamp], 'ACCEL':container_A1[walk_start_stamp:walk_end_stamp]})
    #
    # # 粘合第一段和第二段数据
    # scipy.io.savemat('走路.mat', {'GYRO': container_G1[73000:] + container_G2[:6500],
    #                             'ACCEL': container_A1[73000:] + container_G2[:6500]})
    # scipy.io.savemat('跑步2.mat', {'GYRO': container_G2[6500:],
    #                             'ACCEL': container_A2[6500:]})

    # # 调试绘图
    # plt.plot([sumvector(a[1], a[2], a[3]) for a in container_A1[rope_skipping_start_stamp1:rope_skipping_end_stamp1]])
    # plt.plot([sumvector(a[1], a[2], a[3]) for a in container_G1[rope_skipping_start_stamp1:rope_skipping_end_stamp1]])
    # plt.show()

    # plt.plot([sumvector(a[1], a[2], a[3]) for a in container_A1[rope_skipping_start_stamp2:rope_skipping_end_stamp2]])
    # plt.plot([sumvector(a[1], a[2], a[3]) for a in container_G1[rope_skipping_start_stamp2:rope_skipping_end_stamp2]])
    # plt.show()

    # p1 = [1, 2, 3, 3, 7, 9, 10, 5, 4, 8, 10]
    # p2 = [1, 2, 3, 5, 3, 7, 9, 3, 6, 2, 10, 6]
    # print(dtw(p1, p2))
    '''

    window_size = 5  # s
    frequency = 50  # Hz
    window_length = window_size * frequency
    # tag dict
    tags = {}
    for activity in stamp:
        act = activity
        # act = 'rope_skipping' if re.search('rope_skipping', activity).span() else activity
        tag = label_target_activity(stamp['{}'.format(activity)],
                                    container_A1, window_length=window_length, right_class=act)
        tags.update(tag)

    print(tags)
    # feature dict
    ft = {}
    for k, _ in tags.items():
        ft[k] = get_aw_feature(k, k + window_length, container_G=container_G1, container_A=container_A1)

    print('feature num is:', len(next(iter(ft.values()))[0]))
    print('total dim is:', len(next(iter(ft.values()))))

    # 调试绘图
    # plt.plot([sumvector(a[1], a[2], a[3]) for a in container_A1], label='sum acceleration')
    # plt.plot([sumvector(a[1], a[2], a[3]) for a in container_G1])
    # plt.plot([a[1] for a in container_G1], label='wx')
    # plt.plot([a[2] for a in container_G1], label='wy')
    # plt.plot([a[3] for a in container_G1], label='wz')
    # plt.plot([d[0] for d in delta_angle], label='theta')
    # plt.plot([d[1] for d in delta_angle], label='fai')
    # plt.plot([d[2] for d in delta_angle], label='psi')
    # plt.show()

    # save the feature and tags in pickel file
    pd.DataFrame(ft).to_pickle(r'.\multiclass\feature1_w10.pickle')
    pd.DataFrame(tags, index=[0]).to_pickle(r'.\multiclass\tag1_w10.pickle')
    print('preprocess over')

    # take data and process
    ft = pd.read_pickle(r'multiclass\feature1_w10.pickle')
    tags = pd.read_pickle(r'multiclass\tag1_w10.pickle')
    # ft1 = pd.read_pickle(r'.\multiclass\feature1_w10.pickle')
    # tags1 = pd.read_pickle(r'.\multiclass\tag1_w10.pickle')
    # ft2 = pd.read_pickle(r'.\multiclass\feature_sk_w10.pickle')
    # tags2 = pd.read_pickle(r'.\multiclass\tag_sk_w10.pickle')
    # ft = pd.concat([ft1, ft2], axis=1, sort=False)
    # tags = pd.concat([tags1, tags2], axis=1, sort=False)

    # convert to dataframe with same horizontal index
    df_ft = shuffle(pd.DataFrame(ft).T, random_state=3).T
    df_tags = shuffle(pd.DataFrame(tags, index=[0]).T, random_state=3)

    # ravel the inner part (21d array) of feature child_data matrix
    X = []
    for _, v in df_ft.items():
        temp = []
        for line in v:
            temp.extend(line.tolist())
        X.append(temp)

    Y = df_tags.to_numpy().ravel().tolist()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=17)

    # print(X_train)
    # print(Y_train)
    # print(np.asarray(X_train).shape)
    # print(np.asarray(X_test).shape)
    # print(np.asarray(Y_train).shape)
    # print(np.asarray(Y_test).shape)
    # print(X_test)
    # print(Y_test)

    # clf = svm.SVC(C=10, kernel='rbf', gamma=1.0)
    clf = GaussianNB()

    # Train the model using the training sets
    clf.fit(X_train, Y_train)
    # clf.fit(X_train, Y_train)

    # print(score)

    # cm = confusion_matrix(Y_test, Y_predict)
    # print(cm)
    # print(Y_test)
    # print(Y_predict)

    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    titles_options = [("Confusion matrix, without normalization", None),
                      ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(clf, X_test, Y_test,
                                     cmap='Blues',
                                     normalize=normalize)
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

    plt.show()

    # # calculate scores
    # # score = clf.score(X_test, Y_test)
    # Y_predict = clf.predict(X_test)
    # ns_auc = roc_auc_score(Y_test, Y_predict,  multi_class='ovc')
    # # calculate roc curves
    # ns_fpr, ns_tpr, _ = roc_curve(Y_test, Y_predict)
    # # plot the roc curve for the model
    # plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    # # axis labels
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # # show the legend
    # plt.legend()
    # # show the plot
    # plt.show()
























