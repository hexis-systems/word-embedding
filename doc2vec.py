#!/usr/bin/env python

# http://radimrehurek.com/gensim/models/doc2vec.html
# https://github.com/piskvorky/gensim/blob/develop/docs/notebooks/doc2vec-IMDB.ipynb
# https://www.codatlas.com/github.com/piskvorky/gensim/HEAD/gensim/test/test_doc2vec.py
# cf. https://cs.stanford.edu/~quocle/paragraph_vector.pdf
# cf. http://eng.kifi.com/from-word2vec-to-doc2vec-an-approach-driven-by-chinese-restaurant-process/

from gensim.models.doc2vec import *
import os, sys
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import AffinityPropagation
import matplotlib.pyplot as plt
from itertools import cycle

#from sklearn.mixture import GMM
#import matplotlib as mpl
#import seaborn; seaborn.set()
#import logging
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class getTaggedDocuments(object):
    def __init__(self, dirname):
        self.dirname = dirname
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            if not fname.startswith('.'):
                for line in open(os.path.join(self.dirname, fname)):
                #for line in open(os.path.join(self.dirname, fname), encoding='cp1252'):    # with different encoding
                    yield TaggedDocument(words=line.split(), tags=[fname])


def tsne(model):
    doc_2d = []

    for doc in model.docvecs:
        doc_2d.append(TSNE().fit_transform(doc).tolist()[0])

    return np.asarray(doc_2d)   # return ndarray


def plot_model(doc_2d, labels):
    plt.figure(num=1, figsize=(80, 80), facecolor="w", edgecolor="k")

    for label, doc in zip(labels, doc_2d):
        plt.plot(doc[0], doc[1], ".")
        plt.annotate(label, (doc[0], doc[1]))

    plt.savefig("out_doc2vec.png", dpi=90)
    print("saved output to ./out_doc2vec.png\n")


def plot_gmm(doc_2d, labels, gmm):
    plt.figure(num=1, figsize=(80, 80), facecolor="w", edgecolor="k")

    for label, doc in zip(labels, doc_2d):
        #print(doc, ':', doc[0], ' ', doc[1])
        plt.plot(doc[0], doc[1], ".")
        plt.annotate(label, (doc[0], doc[1]))

    for n, color in enumerate('rgbcy'):
        v, w = np.linalg.eigh(gmm._get_covars()[n][:2, :2])
        #print('v, w: ', v, w)

        print(doc_2d[gmm == n, 0], '-', doc_2d[gmm == n, 1])    # TODO: wo sind die koordinaten in gmm?

        plt.plot(doc_2d[gmm == n, 0], doc_2d[gmm == n, 1], ".", color=color)

        """
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v *= 9
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        """

    plt.savefig("out.png", dpi=90)
    print("saved output to ./out.png\n")

    #sns.jointplot(x="x", y="y", data=doc_2d)


def plot_cluster(af, doc_2d, fnames):
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    n_clusters_ = len(cluster_centers_indices)

    #print(cluster_centers_indices)
    #print(len(af.labels_))
    #print(len(labels))
    #print(n_clusters_)

    plt.figure(num=1, figsize=(80, 80), facecolor="w", edgecolor="k")
    colors = cycle("bgrcmyk")

    for k, col in zip(range(n_clusters_), colors):
        class_members = labels == k         # class_members ist array von boolschen werten, beschreibt cluster membership
        cluster_center = doc_2d[cluster_centers_indices[k]]

        fnames_cluster = []
        fname_indices = [i for i, x in enumerate(class_members) if x]
        for i in fname_indices: fnames_cluster.append(fnames[i])

        #print(fnames_cluster)
        #print(len(class_members))
        #print(len(fnames))
        #print(cluster_center)

        plt.plot(doc_2d[class_members, 0], doc_2d[class_members, 1], col + ".")
        plt.plot(cluster_center[0], cluster_center[1], "o", markerfacecolor=col, markersize=20)

        #plt.annotate(fnames[labels[k]], (cluster_center[0], cluster_center[1]), xytext=(0, -8),
        #        textcoords="offset points", va="center", ha="left")

        for x, fname in zip(doc_2d[class_members], fnames_cluster):
            plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col, linestyle='--', linewidth=1)
            plt.annotate(fname, (x[0], x[1]), xytext=(0, -8),
                        textcoords="offset points", va="center", ha="left")

    plt.savefig("out_doc2vec.png", facecolor="w", dpi=90)
    print("saved output to ./out_doc2vec.png\n")


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("usage: {0} /folder/containing/text/files".format(sys.argv[0]))
        sys.exit(1)


    print('calculating doc2vec model ..\n')

    docs = getTaggedDocuments(sys.argv[1])

    model = Doc2Vec(docs)           # TODO: parameter testen

    labels = list(model.docvecs.doctags.keys())

    print('reducing data ..\n')

    doc_2d = tsne(model)

    print('clustering ..\n')

    af = AffinityPropagation().fit(doc_2d)

    print('plotting ..\n')

    plot_cluster(af, doc_2d, labels)

    #plot_model(doc_2d, labels)

    #gmm = GMM(n_components=5).fit(doc_2d)
    #plot_gmm(doc_2d, labels, gmm)







