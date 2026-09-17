from datasets import load_dataset

from rllm.data.dataset import DatasetRegistry


TRAIN_DATASET_ID = "agentica-org/DeepScaleR-Preview-Dataset"
TEST_DATASET_ID = "HuggingFaceH4/aime_2024"


def prepare_math_data():
    train_dataset = load_dataset(TRAIN_DATASET_ID, split="train")
    test_dataset = load_dataset(TEST_DATASET_ID, split="train")

    def preprocess_fn(example, idx):
        return {
            "question": example["problem"],
            "ground_truth": example["answer"],
            "data_source": "math",
        }

    train_dataset = train_dataset.map(preprocess_fn, with_indices=True)
    test_dataset = test_dataset.map(preprocess_fn, with_indices=True)

    train_dataset = DatasetRegistry.register_dataset("deepscaler_math", train_dataset, "train")
    test_dataset = DatasetRegistry.register_dataset("aime2024", test_dataset, "test")
    return train_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, test_dataset = prepare_math_data()
    print(train_dataset.get_data_path())
    print(test_dataset.get_data_path())
