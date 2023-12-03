# Project of Topological data analysis

# Environment
- gudhi == 3.8.0
- multiprocess == 0.70.15

# Data description
- nonrigid3d: raw dataset

# Distance between data points
- Use *compute_distance.py* to compute geodesic or euclidean distance matrices for dataset  :
    ```
    $ python3 compute_distance --method=geodesic --fdir_out="./processed_geodesic"
    ```

- **processed_euclidean/** and **processed_geodesic/** directories
    - Directories containing processed Euclidean or geodesic distance matrices for each dataset. The file names match the raw data names.
    - The dictionary data within each file includes "dm" (N x N distance matrix for N data points) and "name" (the naprocessed me of the original data).
    - To load the data, use the load_pkl function in tda_tools.py:

    ```
    import tda_tools as tt
    gdata = tt.load_pkl("./processed_geodesic/cat0.pkl")
    ```

# Distance Between Persistence Diagrams
- Use *compute_persim_dist.py* to compute distance betweeen persistent diagram 

Usage example
```
$ python3 compute_persim_dist.py --fdir="./processed_geodesic" --method="bottleneck" --fout="./dist_bottlencek" --nsample=200
```

- Distance matrices for persistence diagrams. Dictionary dataset contains:
    - dist: M x M matrix representing the distance between persistent diagrams, where M is the number of datasets (M=148 for sample data).
    - names: The names of raw data for each dataset.

- To load the data, use the load_pkl function in tda_tools.py:
    ```
    import tda_tools as tt
    gdata = tt.load_pkl("./de_bottleneck.pkl")
    ```

- Exported data
    - **de_bottleneck.pkl**: Bottleneck distance of the persistent diagram computed in Euclidean matrix.
    - **dg_wasserstein_1.pkl**: 1-D Wasserstein distance from geodesic distance.
    - **dg_wasserstein_inf.pkl**: Infinity Wasserstein distance calculated from geodesic distance.



