import pandas as pd
import parameter
import numpy as np
from copy import deepcopy
from datetime import datetime

# def check_discrete_point(times : pd.Series):
#     discrete_point_container = []
#     pre = deepcopy(times[0])
#     for time in times:
#         variance = time - prev
#         if time - prev > 2:
#             discrete_point_container.append(time)
#
#         prev = deepcopy(time)
#     return time


if __name__ == '__main__':
    links = [parameter.kid2_link1, parameter.kid2_link2, parameter.kid2_link3, parameter.kid2_link4]
    for link in links:
        time_mark = link.split('.')[0][-8:]
        if time_mark == '11-48-17':
            print(time_mark)
            log_txt = np.loadtxt(r'C:\Users\20804370\Desktop\activity-rec-feature-extr\activityrecognition\code\python\child2_data\Log.main_sys_2021-01-16__11-48-17.log.txt'.format(time_mark))
            df_log = pd.DataFrame(log_txt)
            print(df_log)

            # time2 = datetime.strptime('12:40:56', '%H:%S:%f').time()
            # time1 = datetime.strptime('12:40:00', '%H:%S:%f').time()
            #
            # df_time['time'] = pd.to_datetime(df_time['time']).dt.time
            # df_time.set_index('time')
            # print(df_time[df_time['time'] > time2])
            # print(df_time['time'])

            # print(df_time[(df_time['time'] > time2) & (df_time['time'] < time1)])
