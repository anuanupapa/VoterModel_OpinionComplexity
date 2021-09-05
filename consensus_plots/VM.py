import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import networkx as nx
import time

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
        nA = np.sum(op[neigh])/(len(op[neigh])+0.0001)
        nB = 1 - nA
        if op[PlInd] == 0 and np.random.random() < (nA)**alphaBA:
            op_next[PlInd] = 1
        elif op[PlInd] == 1 and np.random.random() < (nB)**alphaAB:
            op_next[PlInd] = 0
        else:
            pass
    return(op_next)


def sim(N, p, A, alphaAB, alphaBA):

    [op, AM] = initialize(N, p, A)
    i_main = -1
    while i_main < 200 and np.sum(op) != 0 and np.sum(op) != N:
        i_main = i_main + 1
        op = decide(op, AM, N, alphaAB, alphaBA)

    return(i_main, np.sum(op)/N)


if __name__ == "__main__":

    T = time.time()
    N=100
    p=0.3
    trials=100
    A=0.5
    aBA_arr = np.arange(0,3,0.05)
    aBA_arr = np.round(aBA_arr, 3)
    aAB_arr = np.array([0.5, 0.75, 1., 1.25, 1.5])
    cons_all_arr = np.zeros((len(aAB_arr), len(aBA_arr), trials))
    ctime_all_arr = np.zeros((len(aAB_arr), len(aBA_arr), trials))
    aABInd = -1
    for alphaAB in aAB_arr:
        aABInd = aABInd + 1
        print("start : ", alphaAB)
        aBAInd = -1
        for alphaBA in aBA_arr:
            aBAInd = aBAInd + 1
            print(alphaBA)
            for it in range(trials):
                c_time, consensus = sim(N, p, A, alphaAB, alphaBA)
                ctime_all_arr[aABInd, aBAInd, it] = c_time
                cons_all_arr[aABInd, aBAInd, it] = consensus
            
    np.savez("consensusData_ER0.3.npz", N=N, iniA = A, trials=trials,
             consensus = cons_all_arr, ctime = ctime_all_arr,
             alphaAB = aAB_arr, alphaBA = aBA_arr)
