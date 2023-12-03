import numpy as np
import ripser
import persim
from tqdm import tqdm
from multiprocess import Pool
import tda_tools as tt
import pickle as pkl
from gudhi.wasserstein import wasserstein_distance
import argparse


"""
Usage example

$ python3 compute_persim_dist.py --fdir="./processed_geodesic" --method="bottleneck" --fout="./dist_bottlencek" --nsample=200
"""


_ncore = 20
_nsample = 200
_worder = 1
FDIR = "./processed_geodesic"


def build_arg_parser():
    def get_worder(inp):
        if inp == "inf":
            return np.inf
        else:
            return int(inp)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--fdir", help="data direction", default="./processed_geodesic")
    parser.add_argument("--fout", help="simulated data", required=True)
    parser.add_argument("--worder", help="order for wasserstein distance", default=1, type=get_worder)
    parser.add_argument("--nsample", help="number of the samples", default=200, type=int)
    parser.add_argument("--ncore", help="# of the cores to use for calculation", default=20, type=int)
    parser.add_argument("--method", help="method", choices=["wasserstein", "bottleneck"])
    return parser


def set_args(**kwargs):
    global _ncore, _nsample, _worder, FDIR
    _ncore = kwargs["ncore"]
    _worder = kwargs["worder"]    
    _nsample = kwargs["nsample"]
    FDIR = kwargs["fdir"]


def main(**kwargs):
    
    set_args(**kwargs)
    
    method = kwargs["method"] if _worder < 1000 else "bottleneck"
    fout = kwargs["fout"]
    
    prefix_set = load_prefix()
    
    dgm_set = compute_dgms(prefix_set)
    dist = compute_distance(dgm_set, method)
    save_dist(fout, prefix_set, dist)
    

def parrun(func):
    def wrapper(arg_set):    
        res_all = []
        with tqdm(total=len(arg_set)) as pbar:
            if _ncore == 1:
                for n in range(len(arg_set)):
                    res_all.append(func(arg_set[n]))
                    pbar.update()
            else:
                p = Pool(_ncore)
                for res in p.imap(func, arg_set):
                    res_all.append(res)
                    pbar.update()
                
                p.close()
                p.join()
                    
        return res_all

    return wrapper


def pair_dgms(dgm_set):
    dgms = []
    N = len(dgm_set)
    for i in range(N):
        for j in range(i+1, N):
            for p in range(3): # dimension
                if p == 0:
                    dgm1, dgm2 = dgm_set[i][p][:-1], dgm_set[j][p][:-1]
                else:
                    dgm1, dgm2 = dgm_set[i][p], dgm_set[j][p]
                
                dgms.append([dgm1, dgm2])
    return dgms


def unpair_dist(N, dmax):
    dist = np.zeros([N, N])
    n = 0
    for i in range(N):
        for j in range(i+1, N):
            dist[i, j] = dmax[n]
            dist[j, i] = dist[i, j]
            n += 1

    return dist


def save_dist(fout, prefix_set, dist):
    print("save to %s"%(fout))
    with open(fout, "wb") as fp:
        pkl.dump({
            "names": prefix_set,
            "dist": dist
        }, fp)
        

def compute_distance(dgm_set, method):
    N = len(dgm_set)
    dgms = pair_dgms(dgm_set)
    
    if method == "bottleneck":
        dset = compute_bottleneck(dgms)
        
    elif method == "wasserstein":
        dset = compute_wasserstein(dgms)
    
    else:
        print("method %s not exist"%(method))
        
    dmax = np.max(np.reshape(dset, (-1, 3)), axis=1)
    dist = unpair_dist(N, dmax)
        
        
    return dist


@parrun
def compute_dgms(prefix):
    geom = tt.load_distance(prefix, FDIR, print_warn=False)
    dmat_sub, _ = tt.sample_dmat(geom["dm"], nsample=_nsample)

    res = ripser.ripser(dmat_sub, distance_matrix=True, maxdim=2)
    return res["dgms"]


@parrun
def compute_bottleneck(dgms):
    # dgms = [dgm1, dgm2]
    return persim.bottleneck(dgms[0], dgms[1])


@parrun
def compute_wasserstein(dgms):
    return wasserstein_distance(dgms[0], dgms[1], order=_worder, internal_p=2)


def load_prefix():
    from os import listdir
    
    prefix_set = [f for f in listdir(FDIR) if ".pkl" in f]
    prefix_set.sort()
    
    return prefix_set


if __name__ == "__main__":
    main(**vars(build_arg_parser().parse_args()))
