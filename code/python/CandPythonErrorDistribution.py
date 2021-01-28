# from itertools import izip
from matplotlib import pyplot as plt
for i in range(1, 5):
    error, X, Y = [], [], []
    with open(r"C:\Users\20804370\Desktop\activity-rec-feature-extr\activityrecognition\code\python\C_and_python_results\C{}.txt".format(i), 'r') as cfile, \
            open(r"C:\Users\20804370\Desktop\activity-rec-feature-extr\activityrecognition\code\python\C_and_python_results\P{}.txt".format(i), 'r') as pfile:
            for x, y in zip(cfile, pfile):
                x = x.strip()
                y = y.strip()
                # print("{}      {}".format(x, y))
                error.append(abs(float(x) - float(y)))
                X.append(x)
                Y.append(y)
    plt.bar([i for i in range(1,67)], error)
    plt.show()
    for i, e in enumerate(error):
        if e > 0.01:
            print(i)
            print(X[i], Y[i], e)

    print("*************************************")



