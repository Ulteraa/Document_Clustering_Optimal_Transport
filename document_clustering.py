import ot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.pyplot import legend
from sklearn.cluster import MiniBatchKMeans
from PIL import Image
import pickle
from sklearn.manifold import MDS

def load_data():
    with open('texts.pickle', 'rb') as file:
        texts = pickle.load(file)
    return texts

def costMatrix(i, j, texts):

    X = texts[i][0]
    Y = texts[j][0]

    return ((X[:, None] - Y) ** 2).sum(axis=2)

def clustering(OT_distances):
    embedding = MDS(n_components=2, dissimilarity='precomputed')
    dis = OT_distances - OT_distances[OT_distances > 0].min()
    np.fill_diagonal(dis, 0.)
    embedding = embedding.fit(dis)
    X = embedding.embedding_
    movies = ['DUNKIRK', 'GRAVITY', 'INTERSTELLAR', 'KILL BILL VOL.1', 'KILL BILL VOL.2', 'THE MARTIAN', 'TITANIC']

    plt.figure(figsize=(17, 6))
    plt.scatter(X[:, 0], X[:, 1], alpha=0.)
    plt.axis('equal')
    plt.axis('off')
    c = {'KILL BILL VOL.1': 'red', 'KILL BILL VOL.2': 'red', 'TITANIC': 'blue', 'DUNKIRK': 'blue', 'GRAVITY': 'black',
         'INTERSTELLAR': 'black', 'THE MARTIAN': 'black'}
    for film in movies:
        i = movies.index(film)
        plt.gca().annotate(film, X[i], size=30, ha='center', color=c[film], weight="bold", alpha=0.7)
    plt.show()

def Sinkhorn(a, b, cost, epsilon, max_iter=200):
    K = np.exp(-cost/epsilon)
    v = np.ones(b.shape[0])
    for i in range(max_iter):
        print(K.dot(v))
        u = a/K.dot(v)
        v = b/K.T.dot(u)
    return np.diag(u).dot(K).dot(np.diag(v))


if __name__=='__main__':
    OT_distances = np.zeros((7, 7))
    texts = load_data()
    reg = 0.1
    for i in range(7):
        for j in range(i + 1, 7):
            C = costMatrix(i, j, texts)
            a = texts[i][1]
            b = texts[j][1]
            OT_plan = ot.sinkhorn(a, b, C, reg=reg)
            OT_distances[i, j] = np.sum(C * OT_plan)
            OT_distances[j, i] = OT_distances[i, j]
    clustering(OT_distances)
