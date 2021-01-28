import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import SCORERS, confusion_matrix, \
     plot_confusion_matrix,  plot_roc_curve, roc_curve, \
     roc_auc_score, f1_score, auc, classification_report
from sklearn.preprocessing import label_binarize


# Print confusion matrix
def plot_cf_mat(clf, X_test, Y_test, para):
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    titles_options = [("Confusion matrix for {} by {}, test_ratio={:.3} \n "
                       "window size={}, with {} preprocessing and {} feature_norm".format(para['content'],
                                                                                          para['clf_name'],
                                                                                          para['test_ratio'],
                                                                                          para['window_size'],
                                                                                          para['data_norm'],
                                                                                          para['feature_norm']), None),
                      ("Normalized confusion matrix for {} by {}, test_ratio={:.3} \n"
                       "window size={}, with {} preprocessing and {} feature_norm".format(para['content'],
                                                                                          para['clf_name'],
                                                                                          para['test_ratio'],
                                                                                          para['window_size'],
                                                                                          para['data_norm'],
                                                                                          para['feature_norm']), 'true')]
    # confusion matrix
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(clf, X_test, Y_test,
                                     cmap='Blues',
                                     normalize=normalize)
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)


# plot roc curve in one vs rest style
def plot_roc(Y_test, Y_predict, Y_prob, para):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    Y_test = label_binarize(Y_test, classes=para['n_class'])
    Y_predict = label_binarize(Y_predict, classes=para['n_class'])

    # print(Y_test)
    # print(Y_prob[:, 0])
    # print(roc_auc_score(Y_test, Y_prob, average='micro'))
    plt.figure()
    for j in range(len(para['n_class']) if len(para['n_class']) > 2 else len(para['n_class'])-1):
        if len(para['n_class']) > 2:
            y_true = Y_test[:, j]
            y_prob = Y_prob[:, j]
        else:
            y_true = Y_test[:, 1]
            y_prob = Y_prob[:, 1]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        print(roc_auc_score(y_true, y_prob))

        # plot the roc curve for the model
        plt.plot(fpr, tpr, linestyle='--', label='classes of {}: auc = {}'.format(para['n_class'][j], roc_auc))
        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # show the legend
        plt.legend()
    # plt.title('ROC curve for window_size of {} \n with {} data preprocessing'.format(window_size, norm))

    f, t, _ = roc_curve(Y_test.ravel(), Y_prob.ravel())
    roc_auc = auc(f, t)
    plt.plot(f, t, linestyle='--', label='micro roc: auc = {}'.format(roc_auc))
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    plt.title('ROC curve by {} for {}, test_ratio={:.3} \n'
              'window_size={}, with {} preprocessing and {} feature_norm'.format(para['clf_name'],
                                                                                 para['content'],
                                                                                 para['test_ratio'],
                                                                                 para['window_size'],
                                                                                 para['data_norm'],
                                                                                 para['feature_norm']))
    plt.show()