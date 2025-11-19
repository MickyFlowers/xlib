import io
import json
import multiprocessing as mp
import os
from collections import defaultdict

import h5py
import numpy as np
from PIL import Image


def list_of_dicts_to_dict_of_lists(data_list):
    def merge_dicts(acc, d):
        for k, v in d.items():
            if isinstance(v, dict):
                if k not in acc:
                    acc[k] = {}
                merge_dicts(acc[k], v)
            else:
                if k not in acc:
                    acc[k] = []
                acc[k].append(v)
        return acc

    result = {}
    for d in data_list:
        merge_dicts(result, d)
    return result


def save_episode(group, episode_data: list):
    data_dict = list_of_dicts_to_dict_of_lists(episode_data)
    save_dict_to_hdf5(group, data_dict)


def save_dict_to_hdf5(h5group, data):
    for k, v in data.items():
        if isinstance(v, dict):
            subgroup = h5group.create_group(k)
            save_dict_to_hdf5(subgroup, v)
        elif isinstance(v, (int, float, bool, np.integer, np.floating, np.bool_, np.ndarray)):
            h5group.create_dataset(k, data=v)
        elif isinstance(v, str):
            h5group.create_dataset(k, data=np.string_(json.dumps(v)))
        elif isinstance(v, list):
            if isinstance(v[0], (int, float, bool, np.integer, np.floating, np.bool_, np.ndarray)):
                v_np = np.array(v)
                h5group.create_dataset(k, data=v_np)
            elif isinstance(v[0], bytes):
                vlen_uint8 = h5py.special_dtype(vlen=np.dtype("uint8"))
                dset = h5group.create_dataset(k, (len(v),), dtype=vlen_uint8)
                for i, item in enumerate(v):
                    dset[i] = np.frombuffer(item, dtype="uint8")
            elif isinstance(v[0], str):
                v_str_np = np.array([np.string_(json.dumps(s)) for s in v])
                h5group.create_dataset(k, data=v_str_np)
            else:
                raise ValueError(
                    f"Does not support saving list with non-uniform element types {type(v[0])}."
                )
        else:
            raise ValueError(f"Unsupported data type for key {k}: {type(v)}")


class HDF5Saver(mp.Process):
    def __init__(self, dir_path):
        super().__init__()
        self.queue = mp.Queue()
        self.dir_path = dir_path
        self.stop_flag = mp.Event()
        self.buffer = []

    def add_frame(self, data):
        self.buffer.append(data)

    def save_episode(self):
        self.queue.put(self.buffer)
        self.buffer = []

    def run(self):
        os.makedirs(self.dir_path, exist_ok=True)
        idx = 0
        while not self.stop_flag.is_set() or not self.queue.empty():
            try:
                data = self.queue.get(timeout=0.1)
            except:
                continue
            with h5py.File(os.path.join(self.dir_path, str(idx) + ".hdf5"), "w") as f:
                save_episode(f, data)
                idx += 1

    def stop(self):
        self.stop_flag.set()
