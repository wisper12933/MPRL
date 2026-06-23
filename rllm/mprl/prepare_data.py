from datasets import load_dataset

from rllm.data.dataset import DatasetRegistry

""""
### ALFWorld
"train": "/mnt/home/user28/MPRL/data/indices/alfworld/train.json",
"test": "/mnt/home/user28/MPRL/data/indices/alfworld/test.json",
### SciWorld
"train": "/mnt/home/user28/MPRL/data/indices/sciworld/train.json",
"test": "/mnt/home/user28/MPRL/data/indices/sciworld/test.json",
### WebShop
"train": "/mnt/home/user28/MPRL/data/indices/webshop/train.json",
"test": "/mnt/home/user28/MPRL/data/indices/webshop/test.json",
"""
def prepare_data():
    data_files = {
        "train": "/mnt/home/user28/MPRL/data/indices/webshop/train.json",
        "test": "/mnt/home/user28/MPRL/data/indices/webshop/test.json",
    }
    datasets = load_dataset("json", data_files=data_files)

    train_dataset = DatasetRegistry.register_dataset("webshop", datasets["train"], "train")
    test_dataset = DatasetRegistry.register_dataset("webshop", datasets["test"], "test")
    return train_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, test_dataset = prepare_data()
    print(train_dataset.get_data_path())
    print(test_dataset.get_data_path())
