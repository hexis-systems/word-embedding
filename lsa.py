#!/usr/bin/env python

from gensim import corpora, models
from sklearn.cluster import AffinityPropagation
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np
import os, sys


#import logging
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


##############################################################################
# PARAMETERS

strip_chars = ".,;„“«»‹›–-!?'() "
stopwordlists = []


##############################################################################
# MAIN

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("usage: {0} /folder/containing/text/files".format(sys.argv[0]))
        sys.exit(1)


    ##############################################################################
    # PREPARATION

    # compile stopword list

    string = ""
    stopwords = []

    for file in stopwordlists:
        with open(file, "r") as f:
            stoplist = f.read()
            string += stoplist
            f.close()

    tmp = string.split("\n")
    for word in tmp: stopwords.append(word.strip(strip_chars))
    stopwords = sorted(set(stopwords))


    # load files

    documents = []
    filenames = []

    print("processing ...\n")

    for file in os.listdir(path=sys.argv[1]):
            if not file.startswith("."):
                filenames.append(file)                      # used as plot labels
                file = "".join((sys.argv[1],'/',file))
                print(file)

                with open(file, "r") as f: documents.append(f.read().strip(strip_chars)); f.close()
                #with open(file, "r", encoding='cp1252') as f: documents.append(f.read().strip(strip_chars)); f.close()


    print('\n')

    print("vectorizing ...\n")

    # remove common words and tokenize

    texts = [[word for word in document.lower().split() if word not in stopwords]
             for document in documents]

    # remove words that appear only once

    all_tokens = sum(texts, [])
    tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
    texts = [[word for word in text if word not in tokens_once]
             for text in texts]

    # convert documents to vectors

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]


    ##############################################################################
    # GENERATING MODEL

    model = models.LsiModel(corpus, id2word=dictionary, num_topics=200, chunksize=10) #,
                            #onepass=False, power_iters=6, extra_samples=500)       # sensible: topics 10
                                                                                    # default: topics 200, chunk 2000

    print(model, '\n')


    print("reducing data ...\n")

    matrix_list = []

    for doc in corpus:                                      # querying the model using the original bow document
        tmp_doc = []
        tmp = model.__getitem__(doc) #, eps=0)              # TODO: dokumentieren was __getitem__ hier genau ausgibt
        for word in tmp: tmp_doc.append(word[1])            # save only second part of the tuple
        matrix_list.append(tmp_doc)

    matrix = np.array(matrix_list)                          # in array form

    doc_2d = []
    for doc, file in zip(matrix, filenames):                # reduce the data to 2 dimensions
        #print(file, "\n", doc, "\n\n")    # debug msg
        doc_2d.append(TSNE().fit_transform(doc).tolist()[0])

    matrix = np.asarray(doc_2d)                             # update matrix array


    ##############################################################################
    # CLUSTERING

    print("clustering ...\n")

    #af = AffinityPropagation(damping=0.9, affinity="euclidean", preference=-50).fit(matrix)
    af = AffinityPropagation().fit(matrix)                  # default

    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    n_clusters_ = len(cluster_centers_indices)



    ##############################################################################
    # VISUALIZING

    # cf. http://scikit-learn.org/stable/auto_examples/cluster/plot_affinity_propagation.html
    # annotation cf. http://stackoverflow.com/a/12311782

    """
    plt.figure(num=1, figsize=(80, 80), facecolor="w", edgecolor="k")

    for label, doc in zip(filenames, doc_2d):
        plt.plot(doc[0], doc[1], ".")
        plt.annotate(label, (doc[0], doc[1]))

    plt.savefig("out_lsa.png", dpi=90)
    print("saved output to ./out_lsa.png\n")
    """


    plt.figure(num=1, figsize=(80, 80), facecolor="w", edgecolor="k")
    colors = cycle("bgrcmyk")


    for k, col in zip(range(n_clusters_), colors):
        class_members = labels == k
        cluster_center = matrix[cluster_centers_indices[k]]

        plt.plot(matrix[class_members, 0], matrix[class_members, 1], col + ".")
        plt.plot(cluster_center[0], cluster_center[1], "o", markerfacecolor=col, markersize=20)

        # get doc_id from dimension of original matrix element matching cluster_center[0]
        for i in range(len(matrix_list)):
            if doc_2d[i][0] == float(cluster_center[0]):
                doc_id = i
                break

        plt.annotate(filenames[doc_id], (cluster_center[0], cluster_center[1]), xytext=(0, -8),
                     textcoords="offset points", va="center", ha="left")

        for x in matrix[class_members]:
            plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col, linestyle='--', linewidth=1)

            # get doc_id from dimension of original matrix element matching x[0]
            for i in range(len(matrix_list)):
                if doc_2d[i][0] == float(x[0]):
                    doc_id = i
                    break

            plt.annotate(filenames[doc_id], (x[0], x[1]), xytext=(0, -8),
                         textcoords="offset points", va="center", ha="left")

    plt.savefig("out.png", dpi=90)

    print("saved output to ./out_lsa.png\n")
    