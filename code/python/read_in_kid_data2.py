import pandas as pd
import parameter
from copy import deepcopy
from datetime import datetime
import numpy as np

from feature_extraction.tsfeature import feature_core, feature_fft, feature_time


x = np.fft.fft([1,2,3,4,5])

def get_data(link, time_mark):
    filer = open(
        # first data series
        link,
        'r',
        encoding="UTF-8",
        errors='ignore',
    )
    filew = open(
        '{}.csv'.format(time_mark),
        'w+',
        encoding="UTF-8",
        errors='ignore',
    )
    rline = filer.readline()
    i = 0
    while rline:
        split = rline.split()
        if 'PDR-ACC:' in split:
            # print(split)
            time = split[1]
            ax, ay, az = split[7][:-1], split[8][:-1], split[9][:-1]
            wx, wy, wz = split[12][:-1], split[13][:-1], split[14][:-1]
            # dic = {'time': pd.to_datetime(time).time(), 'ax': ax, 'ay': ay, 'az': az, 'wx': wx, 'wy': wy, 'wz': wz}
            current = '%s,%s,%s,%s,%s,%s,%s\n' % (time, ax, ay, az, wx, wy, wz)
            filew.write(current)
            # print(frame)
            i += 1
            print(i)
        rline = filer.readline()

    filer.close()
    filew.close()


# if __name__ == '__main__':
#     links = [parameter.kid2_link1, parameter.kid2_link2, parameter.kid2_link3, parameter.kid2_link4]
#     # for link in links:
#     #     time_mark = link.split('.')[1][-8:]
#     #     get_data(link, time_mark)
#
#     tag, res = {}, {}
#     # time1 = datetime.strptime('10:59:51.000000', '%H:%M:%S.%f').time()
#     # time2 = datetime.strptime('11:00:35.000000', '%H:%M:%S.%f').time()
#     # tag['girl1_rpsk'] = (time1, time2)
#
#     # time3 = datetime.strptime('12:34:37.000000', '%H:%M:%S.%f').time()
#     # time4 = datetime.strptime('12:35:15.000000', '%H:%M:%S.%f').time()
#     # tag['girl1_situp'] = (time3, time4)
#     #
#     # time5 = datetime.strptime('11:40:02.000000', '%H:%M:%S.%f').time()
#     # time6 = datetime.strptime('11:47:55.000000', '%H:%M:%S.%f').time()
#     # tag['girl1_longhaul'] = (time5, time6)
#
#     people = 'boy2'
#     time1 = datetime.strptime('14:21:43.000000', '%H:%M:%S.%f').time()
#     time2 = datetime.strptime('14:22:30.000000', '%H:%M:%S.%f').time()
#     tag['boy2_rpsk'] = (time1, time2)
#
#     time3 = datetime.strptime('14:23:08.000000', '%H:%M:%S.%f').time()
#     time4 = datetime.strptime('14:23:43.000000', '%H:%M:%S.%f').time()
#     tag['boy2_situp1'] = (time3, time4)
#
#     time5 = datetime.strptime('14:23:55.000000', '%H:%M:%S.%f').time()
#     time6 = datetime.strptime('14:24:23.000000', '%H:%M:%S.%f').time()
#     tag['boy2_situp2'] = (time5, time6)
#
#     time7 = datetime.strptime('14:09:56.000000', '%H:%M:%S.%f').time()
#     time8 = datetime.strptime('14:21:35.000000', '%H:%M:%S.%f').time()
#     tag['boy2_longhaul'] = (time7, time8)
#
#     container_rskp, container_situp1, container_situp2, container_longhaul = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
#     count_rskp, count_situp1, count_situp2, count_longhaul = 0, 0, 0, 0
#     for link in links:
#         time_mark = link.split('.')[1][-8:]
#         print(time_mark)
#         df_data = pd.read_csv(r'C:\Users\20804370\Desktop\activity-rec-feature-extr'
#                               r'\activityrecognition\code\python\child2_data\{}.csv'.format(time_mark),
#                               header=None)
#         df_data.columns = ['time', 'ax', 'ay', 'az', 'wx', 'wy', 'wz']
#         df_data['time'] = pd.to_datetime(df_data['time']).dt.time
#
#         df1_rskp = df_data[(df_data['time'] < time2) & (df_data['time'] > time1)]
#         if(len(df1_rskp)>1):
#             count_rskp += 1
#             container_rskp = df1_rskp
#         df1_situp1 = df_data[(df_data['time'] < time4) & (df_data['time'] > time3)]
#         if (len(df1_situp1) > 1):
#             count_situp1 += 1
#             container_situp1 = df1_situp1
#         df1_situp2 = df_data[(df_data['time'] < time6) & (df_data['time'] > time5)]
#         if (len(df1_situp2) > 1):
#             count_situp2 += 1
#             container_situp2 = df1_situp2
#         df1_longhaul = df_data[(df_data['time'] < time8) & (df_data['time'] > time7)]
#         if (len(df1_longhaul) > 1):
#             count_longhaul += 1
#             container_longhaul = df1_longhaul
#
#     containers = [container_rskp, container_situp1, container_situp2, container_longhaul]
#     counts = [count_rskp, count_situp1, count_situp2, count_longhaul]
#     names = ['rope_skipping', 'situp1', 'situp2', 'longhaul']
#     for count, name in zip(counts, names):
#         print(name)
#         print(count)
#         if count > 1:
#             print('error that time interval overlaps')
#             raise StopIteration
#
#     for name, container in zip(names, containers):
#         container.to_csv("{}{}.csv".format(people, name))

        # df2_rskp = df_data[(df_data['time'] < time2) & (df_data['time'] > time1)]
        # df2_situp1 = df_data[(df_data['time'] < time4) & (df_data['time'] > time3)]
        # df2_situp2 = df_data[(df_data['time'] < time4) & (df_data['time'] > time3)]
        # df2_longhaul = df_data[(df_data['time'] < time6) & (df_data['time'] > time5)]
        #
        # df3_rskp = df_data[(df_data['time'] < time2) & (df_data['time'] > time1)]
        # df3_situp1 = df_data[(df_data['time'] < time4) & (df_data['time'] > time3)]
        # df3_situp2 = df_data[(df_data['time'] < time4) & (df_data['time'] > time3)]
        # df3_longhaul = df_data[(df_data['time'] < time6) & (df_data['time'] > time5)]
        #
        # df4_rskp = df_data[(df_data['time'] < time2) & (df_data['time'] > time1)]
        # df4_situp1 = df_data[(df_data['time'] < time4) & (df_data['time'] > time3)]
        # df4_situp2 = df_data[(df_data['time'] < time4) & (df_data['time'] > time3)]
        # df4_longhaul = df_data[(df_data['time'] < time6) & (df_data['time'] > time5)]

        # if len(df_rskp) > 0 and k in res:
        #     print('error in {}'.format(k))
        #     continue
        # elif len(df_rskp) > 0 and k not in res:
        #     res[k] = df_rskp
        #
        # if len(df_situp) > 0 and k in res:
        #     print('error in {}'.format(k))
        #     continue
        # elif len(df_situp) > 0 and k not in res:
        #     res[k] = df_situp
        #
        # if len(df_longhaul) > 0 and k in res:
        #     print('error in {}'.format(k))
        #     continue
        # elif len(df_longhaul) > 0 and k not in res:
        #     res[k] = df_longhaul







