import gudhi as gd
import numpy as np
import pandas as pd
from persim import plot_diagrams
from TopClassifier import TopClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from gtda.time_series import SingleTakensEmbedding


#f1-micro == accuracy
def accuracy_adjacent(y_true, y_pred):
    conf = np.zeros((9, 9))
    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            for k in range(y_pred.shape[0]):
                conf[i, j] += int(y_true[k] == i + 1 and y_pred[k] == j + 1)

    adjacent_facies = [[1], [0,2], [1], [4], [3,5], [4,6,7], [5,7], [5,6,8], [6,7]]
    acc = 0
    for i in range(conf.shape[0]):
        acc += conf[i, i]
        for j in adjacent_facies[i]:
            acc += conf[i, j]
    return acc / sum(conf.flatten())


data = pd.read_pickle('training_data_final.pkl')

name = ['SHRIMPLIN', 'ALEXANDER D', 'SHANKLE', 'LUKE G U',
        'KIMZEY A', 'CROSS H CATTLE', 'NOLAN', 'Recruit F9',
        'NEWBY', 'CHURCHMAN BIBLE']

#name = ['ALEXANDER D']

features = ['Depth', 'GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'NM_M']


filt_vals = [*range(0, 30)]
maxdims = [1]
modes = ['link', 'star']

accs = pd.DataFrame(columns=['filt_value', 'maxdim', 'mode',
                             'random_label_choice', 'change_data',
                             'mean_accuracy', 'min_accuracy', 'max_accuracy'])

for filt_value in filt_vals:
    for maxdim in maxdims:
        for mode in modes:
            for random_label_choice in [True, False]:
                for change_data in [True, False]: 
                    TC = TopClassifier(filt_value=filt_value,
                                       maxdim=maxdim, mode=mode,
                                       random_label_choice=random_label_choice,
                                       change_data=change_data)
                    avg_acc = 0
                    max_acc = 0
                    min_acc = 1
                    for well in name:
                        df_train = data[data['Well Name'] != well]
                        df_test = data[data['Well Name'] == well]

                        X_train, y_train = np.array(df_train.loc[:, features]), np.array(df_train.loc[:, 'Facies'])
                        X_test, y_test = np.array(df_test.loc[:, features]), np.array(df_test.loc[:, 'Facies'])
                        
                        A = TC.fit(X_train, y_train)
                        y_pred = A.predict(X_test)
                        acc = accuracy_adjacent(y_test, y_pred)
                        max_acc = max(max_acc, acc)
                        min_acc = min(min_acc, acc)
                        avg_acc += acc
                    raw = pd.DataFrame([[filt_value, maxdim, mode,
                                         random_label_choice, change_data, 
                                         np.around(avg_acc / len(name), 3),
                                         np.around(min_acc, 3), 
                                         np.around(max_acc, 3)]],
                                         columns=accs.columns)
                    print(list(raw.loc[0, :]))
                    accs = pd.concat([accs, raw])

accs = accs.reset_index(drop=True)
accs.to_csv('TC_some_shit_results.csv')

