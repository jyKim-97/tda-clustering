import os
import tda_tools as tt
from multiprocess import Pool
from tqdm import tqdm
import pickle as pkl
import argparse


FDIR = None
FDIR_OUT = None
_ncore = 20

"""
Usage example
$ python3 compute_distance --method=geodesic --fdir_out="./processed_geodesic"

"""


def build_arg_parser():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--fdir", help="data direction", default="./nonrigid3d/")
    parser.add_argument("--fdir_out", help="output directory", required=True)
    parser.add_argument("--ncore", help="# of the cores to use for calculation", default=20, type=int)
    parser.add_argument("--method", help="method", choices=["geodesic", "euclidean"])
    return parser


def set_global_args(**kwargs):
    global FDIR, FDIR_OUT, _ncore
    
    FDIR = kwargs["fdir"]
    FDIR_OUT = kwargs["fdir_out"]
    _ncore = kwargs["ncore"]
    

def main(**kwargs):
    
    set_global_args(**kwargs)
    prefix_set = parse_prefix()
    
    method = kwargs["method"]
    if method == "geodesic":
        compute_geodesic(prefix_set)
    else:
        compute_euclidean(prefix_set)
    

def parse_prefix():
    return [f[:-4] for f in os.listdir(FDIR) if ".mat" in f]


def compute_euclidean(prefix_set):
    for prefix in tqdm(prefix_set):
        vert, surf = tt.load_data(os.path.join(FDIR, prefix))
        dmat = tt.compute_euclidean(vert)
        save_dmat(prefix, dmat)


def compute_geodesic(prefix_set):
    
    def _compute(prefix):
        vert, surf = tt.load_data(os.path.join(FDIR, prefix))
        wgraph = tt.gen_wgraph(vert, surf)
        dmat = tt.compute_geodesic(wgraph)
        save_dmat(prefix, dmat)
        return 0
    
    
    with tqdm(total=len(prefix_set), desc="compute geodesic distance") as pbar:
        if _ncore == 1:
            for prefix in prefix_set:
                _compute(prefix)
                pbar.update()
        
        else:
            p = Pool(_ncore)
            for _ in p.imap(_compute, prefix_set):
                pbar.update()
            
            p.close()
            p.join()
    

def save_dmat(prefix, dmat):
    fout = os.path.join(FDIR_OUT, prefix+".pkl") 
    with open(fout, "wb") as fp:
            pkl.dump({
                "name": prefix,
                "dm": dmat
            }, fp)

    
if __name__ == "__main__":
    main(**vars(build_arg_parser().parse_args()))
