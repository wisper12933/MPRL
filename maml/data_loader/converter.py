from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Literal, Union

from ..logger import logging
from .constants import Role, DEFAULT_FORMATTING


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import Seq2SeqTrainingArguments
    
    from ..args import DataArgs
    

logger = logging.get_logger(__name__)

# Attr
## The keys for the Attr is fixed.
@dataclass
class DatasetAttr:
    r"""Dataset attributes."""

    # basic configs
    dataset_file: str
    formatting: Literal["alpaca", "sharegpt"] = "sharegpt"
    # extra configs
    split: str = "train"
    # alpaca columns
    prompt: Optional[str] = "instruction"
    query: Optional[str] = "input"
    response: Optional[str] = "output"
    # sharegpt columns
    messages: Optional[str] = "conversations"
    # sharegpt tags
    role_tag: Optional[str] = "from"
    content_tag: Optional[str] = "value"
    user_tag: Optional[str] = "human"
    assistant_tag: Optional[str] = "gpt"
    system_tag: Optional[str] = "system"

    def __repr__(self) -> str:
        return self.dataset_file


def get_dataset_list(dataset_files: Optional[list[str]]) -> list["DatasetAttr"]:
    r"""Get the attributes of the datasets."""
    if dataset_files is None:
        dataset_files = []

    dataset_list: list[DatasetAttr] = []
    for file in dataset_files:
        dataset_attr = DatasetAttr(dataset_file=file, formatting=DEFAULT_FORMATTING)
        dataset_list.append(dataset_attr)

    return dataset_list

# Converter
@dataclass
class DatasetConverter:
    dataset_attr: "DatasetAttr"
    
    @abstractmethod
    def __call__(self, example: dict[str]) -> dict[str]:
        r"""Convert a single example in the dataset to the standard format."""
        ...
        
## For alpaca formatting
@dataclass
class AlpacaDatasetConverter(DatasetConverter):
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        prompt, query = [], []
        if self.dataset_attr.prompt and example[self.dataset_attr.prompt]:
            query.append(example[self.dataset_attr.prompt])

        if self.dataset_attr.query and example[self.dataset_attr.query]:
            query.append(example[self.dataset_attr.query])

        prompt.append({"role": Role.USER.value, "content": "\n".join(query)})  # "instruction + input"

        if self.dataset_attr.response and isinstance(example[self.dataset_attr.response], str):  # normal example
            response = [{"role": Role.ASSISTANT.value, "content": example[self.dataset_attr.response]}]
        else:  
            raise ValueError("Alpaca output should be str.")
        
        system = ""
        for key in example.keys():
            if "system" in key or "sys" in key:
                system = example[key]
                break

        output = {
            "_prompt": prompt,
            "_response": response,
            "_system": system,
        }
        return output
    
## For sharegpt formatting, preferred
@dataclass
class SharegptDatasetConverter(DatasetConverter):
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        tag_mapping = {
            self.dataset_attr.user_tag: Role.USER.value,
            self.dataset_attr.assistant_tag: Role.ASSISTANT.value,
            self.dataset_attr.system_tag: Role.SYSTEM.value,
        }
        odd_tags = (self.dataset_attr.user_tag)
        even_tags = (self.dataset_attr.assistant_tag)
        accept_tags = (odd_tags, even_tags)
        messages = example[self.dataset_attr.messages]
        if (
            len(messages) != 0
            and messages[0][self.dataset_attr.role_tag] == self.dataset_attr.system_tag
        ):
            system = messages[0][self.dataset_attr.content_tag]
            messages = messages[1:]
        else:
            # try to find system message in other keys
            system = ""
            for key in example.keys():
                if "system" in key or "sys" in key:
                    system = example[key]
                    break

        aligned_messages = []
        broken_data = False
        for turn_idx, message in enumerate(messages):
            if message[self.dataset_attr.role_tag] not in accept_tags[turn_idx % 2]:
                logger.warning_rank0(f"Invalid role tag in {messages}.")
                broken_data = True
                break

            aligned_messages.append(
                {
                    "role": tag_mapping[message[self.dataset_attr.role_tag]],
                    "content": message[self.dataset_attr.content_tag],
                }
            )

        if broken_data:
            logger.warning_rank0("Skipping this abnormal example.")
            prompt, response = [], []
        else:  # normal example
            prompt = aligned_messages[:-1]
            response = aligned_messages[-1:]

        output = {
            "_prompt": prompt,
            "_response": response,
            "_system": system,
        }
        return output

    
DATASET_CONVERTERS = {
    "alpaca": AlpacaDatasetConverter,
    "sharegpt": SharegptDatasetConverter,
}


def register_dataset_converter(name: str, dataset_converter: type["DatasetConverter"]) -> None:
    r"""Register a new dataset converter."""
    if name in DATASET_CONVERTERS:
        raise ValueError(f"Dataset converter {name} already exists.")

    DATASET_CONVERTERS[name] = dataset_converter
    

def get_dataset_converter(name: str, dataset_attr: "DatasetAttr") -> "DatasetConverter":
    r"""Get a dataset converter."""
    if name not in DATASET_CONVERTERS:
        raise ValueError(f"Dataset converter {name} not found.")

    return DATASET_CONVERTERS[name](dataset_attr)


def align_dataset(
    dataset: Union["Dataset", "IterableDataset"],
    dataset_attr: "DatasetAttr",
    data_args: "DataArgs",
    training_args: "Seq2SeqTrainingArguments",
) -> Union["Dataset", "IterableDataset"]:
    r"""Align the dataset to a specific format.

    Aligned dataset:
    _prompt: [{"role": "user", "content": "..."}] * (2T - 1)
    _response: [{"role": "assistant", "content": "..."}] * N (N > 1 for ranking dataset)
    _system: "..."
    """
    column_names = list(next(iter(dataset)).keys())
    kwargs = dict(
        num_proc=16,
        load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
        desc="Converting format of dataset"
    )
    
    dataset_converter = get_dataset_converter(dataset_attr.formatting, dataset_attr)
    return dataset.map(
        dataset_converter,
        batched=False,
        remove_columns=column_names,
        **kwargs
    )