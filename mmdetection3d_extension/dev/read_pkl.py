import pickle
import pprint

pkl_file_path = "/home/wind/Backup/dataset/mmlab_dataset/mmdetection3d/data/pointcloud/pointcloud_dbinfos_train.pkl"

with open(pkl_file_path, "rb") as f:
    data = pickle.load(f)

pprint.pprint(data)
