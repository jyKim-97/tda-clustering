import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import pickle as pkl


_NULL = 10000


def read_dataset(fname, dtype=int):
    data = []
    with open(fname, "r") as fp:
        l = fp.readline()
        while l:
            data.append(list(map(dtype, l[:-1].split())))
            l = fp.readline()
    return np.array(data)


def load_data(prefix):
    vert = read_dataset(prefix+".vert", float)
    surf = read_dataset(prefix+".tri", int)
    return vert, surf


def gen_wgraph(vert, surf):
    
    N = len(vert)

    def find_edge_index(wgraph_single, target):
        for i in range(len(wgraph_single)):
            if target == wgraph_single[i][0]:
                return i
        return -1
    
    wgraph = [[] for _ in range(N)] # edge, weight, counter_id
    for n in range(len(surf)):
        for i in range(3): # triangulation
            id1, id2 = surf[n][i]-1, surf[n][(i+1)%3]-1 # 1-base index
            idx = find_edge_index(np.array(wgraph[id1]), id2)
            
            if idx == -1:
                d = compute_dist(vert[id1], vert[id2])
                wgraph[id1].append([id2, d])
                wgraph[id2].append([id1, d])
    
    return wgraph


def _get_adj(wgraph):
    N = len(wgraph)
    amat = np.zeros((N, N))
    for i in range(N):
        for j in range(len(wgraph[i])):
            id_ = wgraph[i][j][0]
            amat[i, id_] = wgraph[i][j][1]
    return amat


@njit
def _argmin_cond(arr, cond):
    N = len(arr)
    idx = np.arange(N)[cond]
    return idx[np.argmin(arr[cond])]


@njit
def _dijstra(adj_mat):
    N = len(adj_mat)
    dmat = np.ones((N, N)) * _NULL
    
    for n in range(N):
        prev = np.zeros(N)
        dmat_s = np.ones(N) * _NULL
        dmat_s[n] = 0
        
        searching = np.ones(N, dtype=np.int8)
        
        for _ in range(N):
        # while np.any(searching == 1):
            idx_s = _argmin_cond(dmat_s, searching==1)
            searching[idx_s] = 0
            
            for i in range(N):
                if adj_mat[idx_s, i] == 0:
                    continue
                
                dnew = dmat_s[idx_s] + adj_mat[idx_s, i]
                if dnew < dmat_s[i]:
                    dmat_s[i] = dnew
                    prev[i] = idx_s
        
        dmat[n] = dmat_s
        
    return dmat


def compute_geodesic(wgraph):
    adj_mat = _get_adj(wgraph)
    dmat = _dijstra(adj_mat)
    dmat[dmat == _NULL] = -1
    return dmat


# def compute_geodesic(wgraph):
#     from collections import deque
    
#     N = len(wgraph)
#     dmat = np.ones([N, N]) * _NULL
    
#     for id1 in range(N):
#         q = deque([(id1, 0)])
#         q.append((id1, 0))
        
#         used = np.zeros(N, dtype=bool)
#         used[id1] = True
        
#         while q:
#             id2, d = q.popleft()
            
#             if dmat[id1, id2] > d:
#                 dmat[id1, id2] = d
            
#             for i, w in wgraph[id2]:
#                 if used[i]:
#                     continue
#                 used[i] = True  
#                 q.append((i, d+w))
                
#     return dmat


@njit
def compute_euclidean(vert):
    N = len(vert)
    
    dmat = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            dmat[i, j] = compute_dist(vert[i], vert[j])
            dmat[j, i] = dmat[i, j]
        
    return dmat


@njit
def compute_dist(r1, r2):
    return np.sqrt(np.sum((r1 - r2) ** 2))


def farthest_point_sampling(dmat, start_id=0, nsample=200):
    
    id_selected = [start_id]
    d = dmat[start_id]
    for _ in range(1, nsample):
        id_ = np.argmax(d)
        id_selected.append(id_)
        d = np.minimum(d, dmat[id_])
    
    return id_selected


def burning(dmat):
    
    sample_id = 0
    min_null = _NULL
    for i in range(len(dmat)):
        num_null = np.sum(dmat[i] == -1)
        if min_null < num_null:
            sample_id, min_null = i, num_null
    
    id_large = dmat[sample_id] > -1
    id_remove = np.where(~id_large)[0]
    dmat_burn = dmat[:, id_large][id_large]
    
    return dmat_burn, id_remove    


def load_distance(prefix, fdir, print_warn=True):
    from os.path import join
    # from pickle import load
    
    prefix = prefix + ".pkl" if ".pkl" not in prefix else prefix    
    fname = join(fdir, prefix)
    
    geom = _load_dobj(fname)
        
    if np.any(geom["dm"] == _NULL):
        dm, id_r = burning(geom["dm"])
        geom["removed"] = id_r
        geom["dm"] = dm
        
        if print_warn:
            print("Disconnected components exist in %s, removed %d points, %d left"%(fname, len(id_r), len(dm)))
    
    return geom
        

def sample_dmat(dmat, nsample=200, start_id=0):
    id_fps = farthest_point_sampling(dmat, start_id=start_id, nsample=nsample)
    return dmat[id_fps, :][:, id_fps], id_fps


def save_distance(fout, dobj):
    
    @njit
    def extract_triu(dmat):
        N = len(dmat)
        L = N * (N-1) // 2
        triu_flat = np.zeros(L)
        
        n = 0
        for i in range(N):
            for j in range(i+1, N):
                triu_flat[n] = dmat[i, j]
                n += 1
                
        return triu_flat
        
    
    # convert distance matrix structure
    dobj_c = dict(name=dobj["name"])
    dobj_c["dm"] = extract_triu(dobj["dm"]).astype(np.float16)
    
    with open(fout, "wb") as fp:
        pkl.dump(dobj_c, fp)
    

def _load_dobj(fname):
    
    # @njit
    def _construct_full_matrix(triu_flat):
        L = len(triu_flat)
        N = _findN(L)
        
        dmat = np.zeros((N, N))
        n = 0
        for i in range(N):
            for j in range(i+1, N):
                dmat[i, j] = triu_flat[n]
                dmat[j, i] = dmat[i, j]
                n += 1
        
        return dmat
        
    
    # @njit
    def _findN(L):
        N = 1
        L *= 2
        while N * (N-1) < L:
            N += 1
        
        if N * (N-1) != L:
            print("Check L = %d again"%(L))
            N = -1
        
        return N
    
    with open(fname, "rb") as fp:
        dobj = pkl.load(fp)
        dmat = _construct_full_matrix(dobj["dm"].astype(np.float32))
    
    dobj["dm"] = dmat
    return dobj


# def load_pkl(fname):
#     from pickle import load
    
#     with open(fname, "rb") as fp:
#         data = load(fp)
        
#     return data


def _read_obj_name(prefix):
    import re
    return re.findall(r"([a-z]+)\d", prefix)[0]


def divide_boundaries(prefix_set):
    
    prev_obj_name = ""
    yl = [0, len(prefix_set)]
    yt = []
    obj_names = []
    for i in range(len(prefix_set)):
        obj_name = _read_obj_name(prefix_set[i])
        if prev_obj_name != obj_name:
            plt.plot([i, i], yl, 'w--', lw=1)
            plt.plot(yl, [i, i], 'w--', lw=1)
            yt.append(i)
            obj_names.append(obj_name)
            prev_obj_name = obj_name
    plt.xticks(yt)
    plt.yticks(yt, labels=obj_names, rotation=30) 
    plt.ylim(yl)
    plt.xlim(yl)
    
    
def get_true_class(prefix_set):
    true_id = []
    matching = {}
    
    id_ = -1
    prev_obj_name = ""
    for i in range(len(prefix_set)):
        obj_name = _read_obj_name(prefix_set[i])
        if prev_obj_name != obj_name:
            prev_obj_name = obj_name
            id_ += 1
            matching[obj_name] = id_
        
        true_id.append(id_)

    return true_id, matching


# clustering
def sort_matrix(dist_mat, method="single"):
    
    from scipy.spatial.distance import squareform
    from scipy.cluster.hierarchy import linkage
    
    def seriation(linkage_data, cur_index):
        if cur_index < N:
            return [cur_index]
        else:
            left = int(linkage_data[cur_index-N, 0])
            right = int(linkage_data[cur_index-N, 1])
            return (seriation(linkage_data, left) + seriation(linkage_data, right))
    
    N = len(dist_mat)
    square_mat = squareform(dist_mat)
    res_linkage = linkage(square_mat, method=method)
    res_order   = seriation(res_linkage, 2*N-2)
    
    sort_mat = dist_mat.copy()
    sort_mat = sort_mat[res_order, :]
    sort_mat = sort_mat[:, res_order]
    
    return sort_mat, res_linkage, res_order


def hierarchical_clustering(dist_mat, num_clusters, method="single"):
    from sklearn.cluster import AgglomerativeClustering 
    
    aggcluster = AgglomerativeClustering(n_clusters=num_clusters, metric='precomputed', linkage=method)
    cluster_id = aggcluster.fit_predict(dist_mat.copy())
    cluster_id = cluster_id - np.min(cluster_id) + 1

    return cluster_id


def draw_with_dendrogram(res_linkage, cmat, cluster_id, label=None, cth=None):
    from scipy.cluster.hierarchy import dendrogram
    import matplotlib.pyplot as plt
    
    num = res_linkage.shape[0]
    if cth is None:
        cth = max(res_linkage[:, 2])
    
    fig = plt.figure(dpi=120, figsize=(6, 6))
    fig.add_axes([0.15, 0.78, 0.65, 0.2])
    dendrogram(res_linkage, no_labels=True, color_threshold=cth)
    plt.yscale("symlog")
    plt.xticks([]); plt.yticks([])
    # plt.xlim([0, num-1])

    fig.add_axes([0.15, 0.75, 0.65, 0.03])
    plt.imshow(cluster_id.reshape((1, -1)), aspect="auto", cmap="Set3", interpolation="none")
    for cid in np.unique(cluster_id):
        x = np.where(cluster_id == cid)[0]
        plt.text(np.average(x), 0, "%d"%(cid), ha='center', va="center")
    plt.xticks([]); plt.yticks([])
    plt.xlim([0, num-1])

    fig.add_axes([0.15, 0.1, 0.65, 0.65])
    plt.imshow(cmat, cmap="jet")
    plt.xlabel(label, fontsize=14)
    plt.ylabel(label, fontsize=14)

    return fig