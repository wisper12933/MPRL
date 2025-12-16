import os
from typing import TYPE_CHECKING, Union

from datasets import Dataset, DatasetDict, load_dataset, interleave_datasets

from ..logger import logging
from .converter import get_dataset_list, align_dataset
from .processor import MAMLDatasetProcessor, TestDatasetProcessor

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer
    from datasets import Dataset, IterableDataset
    
    from ..args import DataArgs, ModelArgs, TrainingArgs
    from .converter import DatasetAttr
    from .template import Template


FILEEXT2TYPE = {
    "csv": "csv",
    "json": "json",
    "jsonl": "json",
    "txt": "text",
}


logger = logging.get_logger(__name__)


# Load and process datasets
def _load_single_dataset(
    dataset_attr: "DatasetAttr", 
    model_args: "ModelArgs", 
    data_args: "DataArgs", 
    training_args: "TrainingArgs"):
    r"""Load a single dataset and aligns it to the standard format."""
    logger.info_rank0(f"Loading dataset: {dataset_attr}...")
    data_path, data_files = None, []
    local_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_file)
    if os.path.isdir(local_path): # is directory
        for file_name in os.listdir(local_path):
            file_path = os.path.join(local_path, file_name)
            if os.path.isfile(file_path):
                data_files.append(file_path)
    elif os.path.isfile(local_path): # is file
        data_files.append(local_path)
    else:
        raise ValueError("Dataset file not found: {}".format(local_path))
    
    data_path = FILEEXT2TYPE.get(os.path.splitext(data_files[0])[-1][1:], None)
    if data_path is None:
        raise ValueError("Allowed file types: {}.".format(",".join(FILEEXT2TYPE.keys())))
    
    if any(data_path != FILEEXT2TYPE.get(os.path.splitext(data_file)[-1][1:], None) for data_file in data_files):
        raise ValueError("File types should be identical.")

    dataset = load_dataset(
            path=data_path,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=model_args.cache_dir,
            num_proc=data_args.preprocessing_num_workers,
            trust_remote_code=model_args.trust_remote_code,
        )
    
    if data_args.max_samples is not None:
        max_samples = min(len(dataset), data_args.max_samples)
        dataset = dataset.select(range(max_samples))
    
    return align_dataset(dataset, dataset_attr, data_args, training_args)


def _merge_datasets(
    all_datasets: list[Union["Dataset", "IterableDataset"]], data_args: "DataArgs", seed: int
):
    r"""Merge multiple datasets into a unified dataset."""
    if len(all_datasets) == 1:
        return all_datasets[0]
    
    elif data_args.mix_strategy.startswith("interleave"):
        # interleave datasets for meta-training
        return interleave_datasets(
            datasets=all_datasets,
            probabilities=data_args.interleave_probs,
            seed=seed,
            stopping_strategy="first_exhausted" if data_args.mix_strategy.endswith("under") else "all_exhausted",
        )
    
    else:
        raise ValueError(f"Unknown mixing strategy: {data_args.mix_strategy}")


def _get_merged_dataset(
        dataset_files: list[str],
        model_args: "ModelArgs",
        data_args: "DataArgs",
        training_args: "TrainingArgs",):
    r"""Return the merged datasets in the standard format."""
    if dataset_files is None:
        return None
    
    datasets = {}
    # sharegpt formatting is preferred, if using alpaca, specify "formatting" in get_dataset_list
    for dataset_file, dataset_attr in zip(dataset_files, get_dataset_list(dataset_files)):
        datasets[dataset_file] = _load_single_dataset(dataset_attr, model_args, data_args, training_args)
    
    for key, value in datasets.items():
        logger.info_rank0(f"Number of examples in {key}: {len(value)}")
        # print(f"Example from {key}: {value[0]}")
    # removed return_dict
    return _merge_datasets(list(datasets.values()), data_args, seed=training_args.seed)
    

def _get_preprocessed_dataset(
    dataset: Union["Dataset", "IterableDataset"],
    data_args: "DataArgs",
    training_args: "TrainingArgs",
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    is_eval: bool = False,
):
    r"""Preprocesses the dataset, including format checking and tokenization."""
    if dataset is None:
        return None
    
    if is_eval:
        data_processor = TestDatasetProcessor(template, tokenizer, data_args)
    else:
        data_processor = MAMLDatasetProcessor(template, tokenizer, data_args)
        
    column_names = list(next(iter(dataset)).keys())
    kwargs = dict(
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
        desc="Running tokenizer on dataset",
    )
    
    dataset = dataset.map(
        data_processor.preprocess_dataset,
        batched=True,
        batch_size=data_args.preprocessing_batch_size,
        remove_columns=column_names,
        **kwargs,
    )
    
    if training_args.should_log:
        try:
            print("eval example:" if is_eval else "training example:")
            data_processor.print_data_example(next(iter(dataset)))
        except StopIteration:
            raise RuntimeError("Cannot find valid samples, check the data format.")
    
    return dataset


def _split_dataset(
    dataset: Union["Dataset", "IterableDataset"],
    eval_dataset: Union["Dataset", "IterableDataset"],
    data_args: "DataArgs",
    seed: int,
) -> "DatasetDict":
    r"""Split the dataset and returns a dataset dict containing train set and validation set."""
    if eval_dataset is not None and data_args.val_size > 1e-6:
        raise ValueError("Cannot specify `val_size` if `eval_dataset` is not None.")
    
    dataset_dict = {}
    if dataset is not None:
        if data_args.val_size > 1e-6:
            val_size = int(data_args.val_size) if data_args.val_size > 1 else data_args.val_size
            dataset = dataset.train_test_split(test_size=val_size, seed=seed)
            dataset_dict = {"train": dataset["train"], "validation": dataset["test"]}
        else:
            dataset_dict = {"train": dataset}
    
    if eval_dataset is not None:
        dataset_dict["validation"] = eval_dataset
    
    return DatasetDict(dataset_dict)
        

def get_dataset(template: "Template",
                model_args: "ModelArgs",
                data_args: "DataArgs",
                training_args: "TrainingArgs",
                tokenizer: "PreTrainedTokenizer",
) -> dict[str, Union["Dataset", "IterableDataset"]]:
    r"""Get the training dataset"""
    with training_args.main_process_first(desc="load dataset", local=(not data_args.data_shared_file_system)):
        dataset = _get_merged_dataset(data_args.dataset, model_args, data_args, training_args)
        eval_dataset = _get_merged_dataset(data_args.eval_dataset, model_args, data_args, training_args)
        if dataset is not None:
            logger.info_rank0(f"Merge mode: {data_args.mix_strategy}. Total number of examples: {len(dataset)}")
        
    with training_args.main_process_first(desc="pre-process dataset", local=(not data_args.data_shared_file_system)):
        dataset = _get_preprocessed_dataset(dataset, data_args, training_args, template, tokenizer)
        eval_dataset = _get_preprocessed_dataset(eval_dataset, data_args, training_args, template, tokenizer, is_eval=True)
    
        dataset_dict = _split_dataset(dataset, eval_dataset, data_args, training_args.seed)
        
        dataset_module = {}
        if "train" in dataset_dict:
            dataset_module["train_dataset"] = dataset_dict["train"]
        if "validation" in dataset_dict:
            dataset_module["eval_dataset"] = dataset_dict["validation"]
        
        return dataset_module
