"""
Author: Rajdeep Pathak
Date: July 9, 2025
Please set up a proper environment before using this class. It was tested with the RAPIDS 24.12 suite, and requires CuPy, CuDF, Dask, and dask_cudf.
If you use this implementation, we request you to kindly cite our paper:
"""

from tqdm import tqdm
import cudf
import cupy as cp
import dask_cudf
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
from dask import delayed
import gc
import os
import numpy as np
import pandas as pd

class NearestNeighbourDistances:
    """
    Computes the distances between nearest neigbours of one CuDF dataframe with another using the Gower's distance.
    Saves and returns a dataframe (.csv) containing the num_nearest nearest distances. 
    Uses all available GPUs in parallel.
    """
    def __init__(self, real_df, synthetic_df, categorical_columns=None, unique_threshold=None,
                 num_nearest=2, x_chunk_size=50000, y_chunk_size=100000,
                 distances_dir="/tmp/", normalize='minmax', rmm_pool_size='10GB'):
        """
        Initialize the NearestNeighbourDistances calculator.

        Parameters:
        -------------
        - real_df: cudf.DataFrame, the first dataframe
        - synthetic_df: cudf.DataFrame, the second dataframe
        - categorical_columns: list of bools (e.g. [True, False] if the first column is categorical and the second is not) or None
        - unique_threshold: None or int, used only if categorical_columns=None. Determine the categorical columns based on number of unique values
        - num_nearest: int, number of nearest neighbours to keep (default: 2)
        - x_chunk_size: chunk size for the first data (default: 50000, increase/decrease based on GPU memory)
        - y_chunk_size: chunk size for synthetic data (default: 100000, increase/decrease based on GPU memory)
        - distances_dir: str, directory to store intermediate files which will be deleted automatically
        - normalize: None, 'minmax', 'zscore', or 'range' (default: minmax)
        - rmm_pool_size: str, memory to utilize in each GPU: e.g., '15GB' (default: '10GB')
        """
        self.real_df = real_df
        self.synthetic_df = synthetic_df
        self.num_nearest = num_nearest
        self.x_chunk_size = x_chunk_size
        self.y_chunk_size = y_chunk_size
        self.distances_dir = distances_dir
        self.normalize = normalize
        self.rmm_pool_size = rmm_pool_size

        # Flexible handling of categorical columns
        if categorical_columns is not None:
            if not isinstance(categorical_columns, list):
                raise ValueError("categorical_columns must be a list of bools if provided.")
            self.cat_cols = cp.array(categorical_columns)
        else:
            if unique_threshold is None:
                raise ValueError("If categorical_columns is None, you must provide unique_threshold.")
            self.cat_cols = cp.array(self._infer_categorical_columns(unique_threshold))

        if not os.path.exists(distances_dir):
            os.makedirs(distances_dir)

        self.X_size = real_df.shape[0]
        self.Y_size = synthetic_df.shape[0]

    def _infer_categorical_columns(self, unique_threshold):
        """
        Infer categorical columns based on unique value count threshold.
        """
        inferred = []
        for col in self.real_df.columns:
            n_unique = self.real_df[col].nunique()
            if n_unique <= unique_threshold:
                inferred.append(True)
            else:
                inferred.append(False)
        return inferred

    def _preprocess_data(self):
        """Encode categorical columns and normalize numeric columns if required."""

        # Label encode categorical columns
        cat_cols = self.real_df.select_dtypes(include=['object', 'category']).columns

        for col in cat_cols:
            combined = cudf.concat([self.real_df[col], self.synthetic_df[col]]).unique()
            unique_values = combined.to_pandas().tolist()
            mapping = {val: idx for idx, val in enumerate(unique_values)}
            self.real_df[col] = self.real_df[col].map(mapping).astype('int32')
            self.synthetic_df[col] = self.synthetic_df[col].map(mapping).astype('int32')

        # Normalize numerical columns 
        if self.normalize is not None:
            numeric_cols = self.real_df.columns[~cp.asnumpy(self.cat_cols)]

            for col in numeric_cols:
                real_col = self.real_df[col]
                synth_col = self.synthetic_df[col]

                if self.normalize == "minmax":
                    real_min, real_max = real_col.min(), real_col.max()
                    synth_min, synth_max = synth_col.min(), synth_col.max()

                    self.real_df[col] = (real_col - real_min) / (real_max - real_min)
                    self.synthetic_df[col] = (synth_col - synth_min) / (synth_max - synth_min)

                elif self.normalize == "zscore":
                    real_mean, real_std = real_col.mean(), real_col.std()
                    synth_mean, synth_std = synth_col.mean(), synth_col.std()

                    self.real_df[col] = (real_col - real_mean) / real_std
                    self.synthetic_df[col] = (synth_col - synth_mean) / synth_std

                elif self.normalize == "range":
                    real_min, real_max = real_col.min(), real_col.max()
                    synth_min, synth_max = synth_col.min(), synth_col.max()

                    self.real_df[col] = 2 * ((real_col - real_min) / (real_max - real_min)) - 1
                    self.synthetic_df[col] = 2 * ((synth_col - synth_min) / (synth_max - synth_min)) - 1

                else:
                    raise ValueError(f"Unknown normalization type: {self.normalize}")

    def _gower_matrix(self, data_x, data_y):
        """
        Compute Gower distances between two cuDF DataFrames using CuPy.
        They must have the same columns in the same order.
        """

        X = data_x.to_cupy()
        Y = data_y.to_cupy()

        x_n_rows, x_n_cols = X.shape
        y_n_rows = Y.shape[0]

        out = cp.zeros((x_n_rows, y_n_rows), dtype=cp.float32)

        X_cat = X[:, self.cat_cols]
        X_num = X[:, ~self.cat_cols]
        Y_cat = Y[:, self.cat_cols]
        Y_num = Y[:, ~self.cat_cols]

        for i in range(x_n_rows):
            xi_cat = X_cat[i, :]
            xi_num = X_num[i, :]
            xj_cat = Y_cat
            xj_num = Y_num

            sij_cat = cp.where(xi_cat == xj_cat, 0, 1)
            sum_cat = cp.sum(sij_cat, axis=1)

            abs_delta = cp.abs(xi_num - xj_num)
            sum_num = cp.sum(abs_delta, axis=1)

            sums = sum_cat + sum_num
            res = sums / x_n_cols

            out[i, :] = res

        return out

    def _compute_chunk(self, x_chunk, y_partition):
        distances = self._gower_matrix(x_chunk, y_partition)
        distances = cp.partition(distances, self.num_nearest - 1, axis=1)[:, :self.num_nearest]
        return distances

    def compute(self, destination_path="nearest_neighbours.csv"):
        """
        Run the full computation: preprocess, compute distances, save result, and clean up temp files.
    
        Parameters
        ------------
        destination_path (str): Path and filename to save the csv file containing the distances. Default: 'nearest_neighbours.csv'.
        """
    
        self._preprocess_data()
    
        cluster = LocalCUDACluster(rmm_pool_size=self.rmm_pool_size)
        client = Client(cluster)
    
        X_chunks = range(0, self.X_size, self.x_chunk_size)
        Y_chunks = range(0, self.Y_size, self.y_chunk_size)
    
        with tqdm(total=len(X_chunks), desc="Processing the first dataset in chunks", unit="chunk", dynamic_ncols=True) as pbar_x:
            for j in X_chunks:
                x_folder = f"{j//1000}_{(j+self.x_chunk_size)//1000}"
                folder_path = os.path.join(self.distances_dir, x_folder)
                os.makedirs(folder_path, exist_ok=True)
    
                x_chunk = self.real_df.iloc[j:j+self.x_chunk_size]
                scattered_x = client.scatter(x_chunk)
    
                parti = 0
    
                for i in Y_chunks:
                    y_chunk = self.synthetic_df.iloc[i:i+self.y_chunk_size]
                    ddf = dask_cudf.from_cudf(y_chunk, npartitions=10)
    
                    tasks = [
                        delayed(self._compute_chunk)(scattered_x, partition)
                        for partition in ddf.to_delayed()
                    ]
    
                    futures = client.compute(tasks, sync=True)
                    results = client.gather(futures)
                    distances = cp.concatenate(results, axis=1)
                    distances = cp.partition(distances, self.num_nearest - 1, axis=1)[:, :self.num_nearest]
                    distances_df = cudf.DataFrame(distances)
    
                    parti += 1
                    out_file = os.path.join(folder_path, f"{parti}_{i}_{i+self.y_chunk_size}.csv")
                    distances_df.to_csv(out_file, index=False)
    
                    del distances, futures, results, tasks
                    gc.collect()
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()
    
                pbar_x.update(1)
    
        client.close()
        cluster.close()
    
        # Combine all chunks
        combined = []
        folders_to_remove = []
    
        for j in tqdm(X_chunks, desc="Combining results", unit="chunk", dynamic_ncols=True):
            to_hstack = []
            a = int(j / 1000)
            b = int((j + self.x_chunk_size) / 1000)
    
            folder = f"{a}_{b}"
            folders_to_remove.append(folder)
            folder_path = os.path.join(self.distances_dir, folder)
    
            for i in range(1, int(self.Y_size / self.y_chunk_size) + 2):
                file = os.path.join(folder_path, f"{i}_{(i-1)*self.y_chunk_size}_{i*self.y_chunk_size}.csv")
                if os.path.exists(file):
                    df = pd.read_csv(file)
                    to_hstack.append(df)
    
            if to_hstack:
                temp = np.concatenate(to_hstack, axis=1)
                temp = np.partition(temp, self.num_nearest - 1, axis=1)[:, :self.num_nearest]
                combined.append(temp)
    
        final_results = np.concatenate(combined, axis=0)
        columns = [f"nearest_{i+1}" for i in range(self.num_nearest)]
        final_df = pd.DataFrame(final_results, columns=columns)
        final_df.to_csv(destination_path, index=False)
    
        # Clean up the temporary folders
        for folder in folders_to_remove:
            folder_path = os.path.join(self.distances_dir, folder)
            if os.path.exists(folder_path):
                for f in os.listdir(folder_path):
                    os.remove(os.path.join(folder_path, f))
                os.rmdir(folder_path)
    
        return final_df
