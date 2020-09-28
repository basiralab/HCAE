import numpy as np
import settings
import tensorflow as tf
from sklearn import svm
from sklearn.model_selection import train_test_split
from encoder import Encoder
from hypergraph_utils import construct_H_with_KNN, generate_G_from_H

flags = tf.app.flags
FLAGS = flags.FLAGS

"""Main function of HyperConnectome AutoEncoder(HCAE) for training and predicting
    HCAE(samples, labels, view)
    Inputs:
        view: An integer higher or equal to zero that selects what single view will the data samples be taken from
              -1 for fusing all the views together
        labels: (n x 1) matrix containing the labels of the data samples [0, 1]
                 n is the number of samples in the dataset
        samples: if view is different from -1 (multi-view):
			(n x m x m) matrix of original data
			n is the number of samples in dataset
                        m is the number of nodes
		if multi-view:
			(l x n x m x m) original data
			l is the number of views (always 4 for our data)
                        n is the number of samples in dataset
                        m is the number of nodes

    Outputs:
        result: (n x m x 1) matrix containing the extracted embeddings
                n the number of samples in the dataset
                m the number of nodes
        mean_err: the mean of classification error over the samples of the test set
        std_err:  the standard deviation of classification error over the samples of the test set
        avg_cost: mean of reconstruction error of the samples of the dataset
        
    Feel free to modify the hyperparameters in the settings.py file according to your needs"""


def HCAE(samples, labels, view=0):
    model = 'hyper_arga_ae'
    settings1 = settings.get_settings_new(model, view)

    enc = Encoder(settings1)
    if view == -1:
        adjacency0 = samples[0]
        adjacency1 = samples[1]
        adjacency2 = samples[2]
        adjacency3 = samples[3]
    else:
        adjacency0 = samples

    subject = adjacency0.shape[0]
    n = adjacency0.shape[1]

    result = np.zeros((subject, n, 1))

    H = []
    G = []

    for i in range(0, subject):
        if view == -1:
            H.append(np.concatenate((adjacency0[i], adjacency1[i], adjacency2[i], adjacency3[i]), axis=1))
        else:
            H.append(adjacency0[i])

    hypergraphs = []
    for i in range(subject):
        if view == -1:
            hypergraph0 = construct_H_with_KNN(adjacency0[i], K_neigs=FLAGS.multi_view_K)  # optimal value for multi-view is 13
            hypergraph1 = construct_H_with_KNN(adjacency1[i], K_neigs=FLAGS.multi_view_K)
            hypergraph2 = construct_H_with_KNN(adjacency2[i], K_neigs=FLAGS.multi_view_K)
            hypergraph3 = construct_H_with_KNN(adjacency3[i], K_neigs=FLAGS.multi_view_K)
            hypergraph = np.concatenate((hypergraph0, hypergraph1, hypergraph2, hypergraph3), axis=1)
        else:
            hypergraph = construct_H_with_KNN(H[i], K_neigs=FLAGS.single_view_K)  # optimal values for our single views 7,8,11,9,13
        hypergraphs.append(hypergraph)
    hypergraphs = np.asarray(hypergraphs)

    for i in range(0, subject):
        G.append(generate_G_from_H(hypergraphs[i]))

    G = np.asarray(G)

    for i in range(0, subject):
        print(' ')
        print('Subject: ' + str(i + 1))
        encoded_view = enc.erun(H[i], G[i])
        result[i] = encoded_view

    avg_cost = enc.cost / subject
    print('Avg cost: ' + str(avg_cost))

    X = np.zeros((subject, n))

    for i in range(result.shape[0]):
        X[i] = result[i].transpose()

    accuracies = []
    for epoch in range(0, 100):
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, shuffle=True)
        clf = svm.SVC(gamma='scale')
        clf.fit(X_train, y_train.ravel())
        t = clf.predict(X_test)
        result = 0
        y_test = y_test.ravel()
        t = t.transpose()
        for i in range(y_test.shape[0]):
            result += (y_test[i] == t[i])

        accuracy = float(result) / y_test.shape[0]
        accuracies.append(accuracy)

    mean_err = np.mean(np.asarray(accuracies))
    std_err = np.std(np.asarray(accuracies))
    print('------------------')
    print('HCAE')
    print('Mean: ' + str(mean_err * 100) + '%')
    print('Std: ' + str(std_err * 100) + '%')
    print(' ')

    return result, mean_err, std_err, avg_cost

