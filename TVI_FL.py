# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 15:09:41 2020

@author: Anonymous

License:
This code and the accompanying files is a preliminary release that has the exclusive purpose to facilitate the review process of the paper entitled "Learning the piece-wise constant graph structure of a varying Ising model" for publication in the ICML 2020. The code will be further polished to improve its utility and user-friendliness, and then will become available online after the publication.

Please do not distribute to others or use this work for other purpose.

Read the README.txt file for more information.

"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
import itertools


def TVI_FL(X, lambda_1, lambda_2):
    '''
                            TVI-FL algorithm

    Inputs:

    - X: List of n ndarray of shape (p * n_i)
    - lambda_1: Hyperparameter related to the fused penalty
    - lambda_2: Hyperparameter related to the lasso penalty

    Outputs:

    - Graph: ndrray of shape (n * p * p) containing the n learned graphs
    - ChangePoints: List of change-point indices

    '''

    p = X[0].shape[0]
    n = len(X)

    Graph = np.zeros((n, p, p))

    CP = []

    for node in range(p):

        print('Learning neighborhood of node number ' +
              str(node + 1) + '/' + str(p) + '...')

        # For each node, learn its neighborhood using the corresponding
        # function
        beta = neihgborhood(X, lambda_1, lambda_2, node)

        cpp = np.where(np.round(np.linalg.norm(
            beta[:, 1:] - beta[:, :-1], 2, axis=0), 4) > 0)[0]      # Consider change-point if difference below 10e-4
        CP = CP + [cpp]

        if node > 0:
            Graph[:, node, :node] = beta[:node, :].T
        if node < p - 1:
            Graph[:, node, node + 1:] = beta[node:, :].T

    ChangePoints = np.unique(list(itertools.chain.from_iterable(CP)))

    return Graph, ChangePoints


def plot_graph(vote, layout, graph_ts, info_party):
    ''' Function to plot US graph '''

    i = vote
    pos = layout
    GG = graph_ts
    fix = layout.copy()
    del fix[9]

    plt.figure()

    W = abs((GG[i] > 0) * GG[i])
    W = (W + W.T) * .5
    G = nx.from_numpy_matrix(W)

    pos = nx.spring_layout(G, pos=pos)

    nx.draw_networkx_nodes(G, pos, nodelist=list(np.where(info_party == 'rep')[
                           0]), node_color='r', alpha=0.7, node_shape='o')  # print republican
    nx.draw_networkx_nodes(G, pos, nodelist=list(np.where(info_party == 'dem')[
                           0]), node_color='b', alpha=0.7, node_shape='o')  # print democrat
    nx.draw_networkx_nodes(
        G, pos, nodelist=[9], node_color='purple', alpha=0.7, node_shape='o')

    weight_val = [d['weight'] * 7 for (u, v, d) in G.edges(data=True)]

    nx.draw_networkx_edges(G, pos, width=weight_val)

    labels = {k: k + 1 for k in np.arange(0, 18)}
    nx.draw_networkx_labels(G, pos, labels, font_size=10,
                            font_color='w', font_weight='bold')


def neihgborhood(X, lambd_1, lambd_2, node):
    '''
                Neighborhood selection using CVXPY

    Inputs:

    - X: List of n ndarray of shape (p * n_i)
    - lambda_1: Hyperparameter related to the fused penalty
    - lambda_2: Hyperparameter related to the lasso penalty
    - node: Considered node

    Output:

    - beta: ndrray of shape ((p-1) * n) containing the n learned neighborhood
            of Node

    '''

    n = len(X)
    p = X[0].shape[0]
    beta = cp.Variable((p - 1, n))

    not_a = list(range(p))
    del not_a[node]

    log_lik = 0         # Construction of the objective function
    for i in range(n):

        n_i = X[i].shape[1]

        blob = beta[:, i] @ X[i][not_a, :]

        log_lik += (1 / n_i) * cp.sum(
            -cp.reshape(cp.multiply(X[i][node, :], blob), (n_i,)) +
            cp.log_sum_exp(
                cp.hstack([- cp.reshape(blob, (n_i, 1)), cp.reshape(blob, (n_i, 1))]), axis=1)
        )

    l1 = cp.Parameter(nonneg=True)
    l2 = cp.Parameter(nonneg=True)

    reg = l2 * cp.norm(beta, p=1) + l1 * \
        cp.sum(cp.norm(beta[:, 1:] - beta[:, :-1],
                       p=2, axis=0))  # Penalty function

    function = 0.01 * log_lik + reg             # Divide by 100 for numerical issues
    problem = cp.Problem(cp.Minimize(function))

    l1.value = lambd_1
    l2.value = lambd_2
    problem.solve(solver=cp.ECOS, verbose=False)  # Solve problem

    beta = np.round(beta.value, 5)

    return beta
