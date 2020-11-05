import numpy as np


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A):  # 除以每列的和
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD


def get_adj_sym_matrix(skeleton, norm=False):
    num_joints = skeleton.get_num_joints()
    self_link = [(i, i) for i in range(num_joints)]

    # adjacency pairs
    inward = skeleton.get_pairs()
    outward = [(j, i) for i, j in inward]
    neighbor = inward + outward

    # symmetry pairs
    symmetry_up = skeleton.get_symmetry()
    symmetry_low = [(j, i) for i, j in symmetry_up]
    symmetry = symmetry_up + symmetry_low

    adj_graph = edge2mat(self_link+neighbor, num_joints)
    sym_graph = edge2mat(self_link+symmetry, num_joints)

    if norm:
        adj_graph = normalize_undigraph(adj_graph)
        sym_graph = normalize_undigraph(sym_graph)

    return adj_graph, sym_graph


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os
    from tools.skeleton import Skeleton

    # os.environ['DISPLAY'] = 'localhost:11.0'
    ntu_skeleton = Skeleton(mode='GAST60')
    A = get_adj_sym_matrix(ntu_skeleton, norm=True)
    for i in A:
        plt.imshow(i, cmap='gray')
        plt.show()
    print(A[0])