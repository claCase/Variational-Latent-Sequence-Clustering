from tqdm import tqdm
import tensorflow as tf 
import string
import pickle
from typing import Optional, Tuple, List, Dict
import pandas as pd
import zipfile as zf
import os
import numpy as np
import argparse


def process_motion_sense(dataset_path, save_path):
    _, _, files = next(os.walk(dataset_path))
    # print(files)
    subjects = None
    max_len = 0
    f = files[0]
    if f[-3:] == "zip":
        f_path = os.path.join(dataset_path, f)
        zipped = zf.ZipFile(f_path)
        files_in_zip = zipped.namelist()
        files_in_zip.sort()
        for fz in files_in_zip:
            if fz[-3:] == "csv" and fz[:2] != "__":
                subject = fz.split("_")[-1].split(".")[0]
                fzo = zipped.open(fz)
                print(f"folder: {f}  file: {fz}  subject: {subject}")
                df = pd.read_csv(fzo, delimiter=",", header=0, index_col=0)
                tmove = [fz.split("/")[1]] * len(df)
                df["movement"] = tmove
                df["idx"] = np.asarray(range(len(df)), dtype=str)
                df["subject"] = [subject] * len(df)
                df.set_index(["subject", "movement", "idx"], inplace=True)
                if subjects is None:
                    subjects = df
                else:
                    subjects = pd.concat([subjects, df], axis=0)
    subjects.index = subjects.index.set_levels(
        [*subjects.index.levels[:-1], subjects.index.levels[-1].astype(int)]
    )
    if save_path is not None:
        subjects.to_csv(os.path.join(save_path, "motion_sense.csv"))
    return subjects

def load_motion_sense(path):
    df = pd.read_csv(path)
    df.set_index(['subject', 'movement', 'idx'], inplace=True)
    return df 


def df_to_matrix(df: pd.DataFrame,
                 idx_mapping=None,
                 save_path: os.path = None,
                 name: str = "default") -> Tuple[np.ndarray, list[dict], list]:
    print("DataFrame to Matrix conversion")
    idx = df.index
    levels = idx.nlevels
    idx_np = np.array((*zip(idx.tolist()),)).squeeze()
    unique_idx = [np.sort(np.unique(idx_np[:, i])) for i in range(levels)]
    levels_name = list(idx.names)
    if idx_mapping is None:
        print("Processing indices...")
        unique_range = [np.arange(len(i)) for i in unique_idx]
        max_ids = [i[-1] + 1 for i in unique_range]
        maps = [{} for _ in range(levels)]
        for l in range(levels):
            for i, j in tqdm(zip(unique_idx[l], unique_range[l]), total=len(unique_range[l])):
                maps[l][i] = j
        matrix = np.zeros(shape=(*max_ids, len(df.columns)))
    else:
        assert len(idx_mapping) == levels
        maps = idx_mapping
        max_ids = [np.max(list(i.values())) for i in maps]

    print("Assigning values to indices...")
    for id, r in tqdm(zip(idx_np, df.iterrows()), total=len(idx_np)):
        np_ids = [maps[i][j] for i, j in enumerate(id)]
        matrix[(*np_ids, None)] = r[1:]

    columns = list(df.columns)

    if save_path:
        try:
            np.save(os.path.join(save_path, name + "_matrix.npy"), matrix)
            np.save(os.path.join(save_path, name + "_unique_index.npy"), unique_idx)
            with open(os.path.join(save_path, name + "_index_mappings.pkl"), "wb") as file:
                pickle.dump(maps, file)
            with open(os.path.join(save_path, name + "_columns_names.pkl"), "wb") as file:
                pickle.dump(columns, file)
            with open(os.path.join(save_path, name + "_levels_names.pkl"), "wb") as file:
                pickle.dump(levels_name, file)
        except Exception as e:
            raise e
    return matrix, maps, columns, levels_name


def matrix_to_df(matrix: np.ndarray, idx_mapping: List[Dict], columns_name=None, levels_name=None, **kwargs):
    sparse_matrix = tf.sparse.from_dense(matrix)
    idx = sparse_matrix.indices.numpy()
    column_names = [string.ascii_letters[i] for i in range(matrix.shape[-1])]
    columns = [column_names[i[-1]] for i in idx]
    values_idx = []
    for mapping in idx_mapping:
        values_idx.append(dict(zip(idx_mapping[0].values(), idx_mapping[0].keys())))

    idx_list = [(*v,) for v in zip(*values_idx)]

    multiindex = pd.MultiIndex.from_tuples(idx_list, names=levels_name)
    df = pd.DataFrame(sparse_matrix.values.numpy(), index=multiindex)
    df = df.unstack(-1).droplevel(0, 1)
    if columns_name is not None:
        df.columns = columns_name
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-path")
    args = parser.parse_args()
    path = args.path

    df = process_motion_sense(path, "./assets/df.csv")
