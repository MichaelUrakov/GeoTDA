import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score

class TopClassifier:

    #Инициализация модели

    def __init__(self, filt_value=1, maxdim=1, mode='link', random_label_choice=False, change_data=False):
        self.filt_value = filt_value # значение фильтрации
        self.maxdim = maxdim # максимальная размерность симплексов в комплексе
        self.mode = mode # линк или звезда
        self.random_label_choice = random_label_choice # случайный выбор метки
        self.change_data = change_data # изменять датасет

    #Обучение модели

    def fit(self, X, y):
        self.pointcloud = X # облако точек
        self.labels = y # метки этих точек
        if X.shape[0] != y.shape[0]:
            return '\nError: разный размер X и y'
        self.labels_dict = {np.unique(y)[i]: i for i in range(len(np.unique(y)))} # набор всех меток
        return self

    #Возвращает параметры модели

    def get_params(self, list_type=False):
        if not list_type:
            return {'filt_value': self.filt_value,
                    'maxdim': self.maxdim,
                    'mode': self.mode,
                    'random_label_choice': self.random_label_choice,
                    'change_data': self.change_data}
        else:
            return [self.filt_value, self.maxdim, self.mode, self.random_label_choice, self.change_data]

    #Возвращает звезду заданного симплекса

    def get_star(self, simp):
        return [i[0] for i in self.filt.get_star(simp)]

    #Возвращает линк заданного симплекса

    def get_link(self, simp):
        simp = sorted(simp)
        star = [i[0] for i in self.filt.get_star(simp)]
        link = [sorted(list(set(i).difference(set(simp)))) for i in star] # линк можно получить из звезды
        if len(link) != 0:
            link.remove([])
        return link

    #Предсказывает метки заданных точек

    def predict(self, X):
        #облако точек
        self.pointcloud = np.append(self.pointcloud, X, axis=0)
        #VR-комплекс на этих точках
        self.filt = gd.RipsComplex(self.pointcloud,
                                   max_edge_length=self.filt_value).create_simplex_tree(max_dimension=self.maxdim)
        
        prediction = []
        # Ассоциирующая функция
        def ass_func(simplex):
            res = np.zeros((len(self.labels_dict), 1))
            for vertex in simplex:
                if len(self.labels) > vertex:
                    res[self.labels_dict[self.labels[vertex]]] += 1
            return res    

        # Функция Расширения
        def ext_func(vertex):
            res = np.zeros((len(self.labels_dict), 1))
            for simp in int(self.mode == 'link') * self.get_link(vertex) + int(self.mode == 'star') * self.get_star(vertex):
                e = self.filt.filtration(simp) + int(self.filt.filtration(simp) == 0) * 0.000001 # значение фильтрации симплекса
                res += ass_func(simp) / e
            return res
        
        for v in range(self.pointcloud.shape[0] - X.shape[0], self.pointcloud.shape[0]):
            ext = ext_func([v]) # предсказываем веса меток для данной вершины
            if self.random_label_choice:
                max_ext = max(ext)
                label = list(self.labels_dict.keys())[np.random.choice([i for i in range(len(ext)) if (ext[i] == max_ext)])] # случайно выбираем из максимальных меток
                prediction.append(label)
            else:
                label = list(self.labels_dict.keys())[np.argmax(ext)] #выбираем первую максимальную метку
                prediction.append(label)
            if self.change_data == True:
                # обновляем облако точек и список их меток
                self.labels = np.append(self.labels, np.array(label))
        return np.array(prediction)

    # Возвращает оценку предсказания, вычисленной по функции score_func

    def score(self, X, y, score_func=accuracy_score):
        return score_func(y, self.predict(X))
    
'''
np.random.seed(0)
fig, ax = plt.subplots()
fig.set_size_inches((7, 7))
ax.set_ylim(-1, 1)
ax.set_xlim(-1, 1)

n = 10 # кол-во точек в кружочке

# определяем облако точек
X = 0.1 * np.random.random((n, 2)) + [-0.05, 0.607]
X = np.append(X, 0.1 * np.random.random((n, 2)) + [0.5, -0.5], axis=0)
X = np.append(X, 0.1 * np.random.random((n, 2)) + [-0.55, -0.5], axis=0)

# определяем метки точек
labels = np.array([0, 1, 2]).astype(np.float64)
y = np.array(np.array([[i] * n for i in labels]).flatten())

for i in range(len(labels)):
    ax.scatter(X[np.where(y == i)][:, 0], X[np.where(y == i)][:, 1], color=['r', 'g', 'b'][i])

# задаем точки, которые будем классифицировать
t = np.linspace(0, 2*np.pi, num=30)
circle = np.array([0.0 + 0.3 * np.cos(t), 0.0 + 0.3 * np.sin(t)]).T

test = circle #np.concatenate([[[x, y] for x in np.linspace(-1, 1, num=20)] for y in np.linspace(-1, 1, num=20)])
#ax.scatter(test[:, 0], test[:, 1], color='k') 

# параметры модели
filt_value = 0.6
maxdim = 1
mode = 'link'
random_label_choice = False
change_data = False

# задаем модель
TC = TopClassifier(filt_value=filt_value, maxdim=maxdim, mode=mode,
                  random_label_choice=random_label_choice,
                  change_data=change_data)

# обучаем модель
A = TC.fit(X, y)

# строим предсказание 
pred = A.predict(test)

for i in range(len(test)):
    label = pred[i]
    ax.scatter(test[i, 0], test[i, 1], color=['r', 'g', 'b'][int(label)])

ax.axis('off')
'''
