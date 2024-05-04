import ast
import logging

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from src.data_utils import get_encoded_prompt


class LLMSotaDataset(Dataset):
    """Dataset for prompting with LLMs like Orca or LlaMa for SotA tasks (loading data from txt file)."""

    def __init__(
            self,
            data_path: str,
            instruction: str,
            tokenizer: PreTrainedTokenizerBase,
            max_data: int = 0,
            instruction_tuning: bool = False,
            testing: bool = False,
    ):
        self._data_path = data_path
        self._tokenizer = tokenizer
        self._max_data = max_data
        self._instruction = instruction
        self._instruction_tuning = instruction_tuning
        self._testing = testing

        self._labels = []
        self._texts = []
        self._encoded_inputs = []
        self._instruction_tuning_prompts = []

        self._load_data()

    def __len__(self) -> int:
        """
        Return the length of the dataset.

        :return: length of the dataset
        """
        return len(self._texts)

    def __getitem__(self, index: int) -> dict:
        if self._instruction_tuning and not self._testing:
            return {
                "input_ids": self._encoded_inputs[index]["input_ids"].squeeze(),
                "attention_mask": self._encoded_inputs[index]["attention_mask"].squeeze(),
            }

        return {
            "input_ids": self._encoded_inputs[index]["input_ids"].squeeze(),
            "attention_mask": self._encoded_inputs[index]["attention_mask"].squeeze(),
            "text": self._texts[index],
            "labels": self._labels[index],
        }

    def _load_data(self) -> None:
        with open(self._data_path, "r", encoding="UTF-8") as file:
            for line in file:
                line = line.strip()
                if line == "":
                    continue
                text, labels = line.split("####")
                # replace all NULL with null in labels
                labels = labels.replace("NULL", "null")
                labels = ast.literal_eval(labels)
                # if the labels is list of lists, convert it to list of tuples
                if isinstance(labels[0], list):
                    labels = [tuple(label) for label in labels]

                self._labels.append(labels)
                self._texts.append(text)

                encoded_input = get_encoded_prompt(
                    labels=labels,
                    text=text,
                    tokenizer=self._tokenizer,
                    instruction=self._instruction,
                    instruction_tuning=self._instruction_tuning,
                    testing=self._testing,
                )

                self._encoded_inputs.append(encoded_input)

                if 0 < self._max_data <= len(self._texts):
                    break

        logging.info("Example of first label: %s", str(self._labels[0]))
        logging.info("Example of first text: %s", str(self._texts[0]))
        logging.info(
            "Example of first prompt: %s", str(self._tokenizer.decode(self._encoded_inputs[0]["input_ids"][0]))
        )
        logging.info("Number of samples: %d", len(self._texts))
