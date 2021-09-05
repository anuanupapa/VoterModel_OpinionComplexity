import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import networkx as nx

def init_adjmat(N, p):
    G = nx.generators.random_graphs.erdos_renyi_graph(N, p)
    adjacency_matrix = nx.to_numpy_matrix(G)
    return(adjacency_matrix)


def initialize(N, p, A):
    AdjMat = init_adjmat(N, p)
    opinion = np.zeros((N, 1))
    for i in range(N):
        if np.random.random()<A:
            opinion[i] = 1
    return(opinion, AdjMat)

@nb.njit
def decide(op, AM, N, alphaAB, alphaBA):

    op_next = op.copy()
    for PlInd in range(N):
        neigh = np.where(AM[PlInd] == 1)[0]
        A = np.sum(op[neigh])/len(op[neigh])
        B = 1 - A
        if op[PlInd] == 0 and np.random.random() < (A/N)**alphaBA:
            op_next[PlInd] = 1
        elif op[PlInd] == 1 and np.random.random() < (B/N)**alphaAB:
            op_next[PlInd] = 0
        else:
            pass
    return(op_next)


def sim(N, p, A, alphaAB, alphaBA):

    A_arr = []
    rounds = []
    
    [op, AM] = initialize(N, p, A)
    i_main = -1
    while i_main < 100000 and np.sum(op) != 0 and np.sum(op) != N:
        i_main = i_main + 1
        op = decide(op, AM, N, alphaAB, alphaBA)
        A_arr.append(np.sum(op)/N)
        rounds.append(i_main)

    return(A_arr, rounds)


if __name__ == "__main__":
    [A, t] = sim(100, 0.3, 0.5, 0.7, 1.)
    plt.plot(t, A, 'o-')
    plt.show()
