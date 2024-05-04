from transformers import PreTrainedTokenizerBase, BatchEncoding


def get_encoded_prompt(
        labels: list[tuple[str]],
        text: str,
        tokenizer: PreTrainedTokenizerBase,
        instruction: str,
        instruction_tuning: bool,
        testing: bool,
) -> BatchEncoding:
    chat = [
        {"role": "system", "content": "You are an aspect-based sentiment analysis classifier."},
        {"role": "user", "content": f"{instruction}\nInput: \"\"\"{text}\"\"\""},
    ]
    if instruction_tuning and not testing:
        template = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        str_labels = "[" + ", ".join("(" + ", ".join(f'"{item}"' for item in label) + ")" for label in labels) + "]"
        template_with_labels = f'{template}Sentiment elements: {str_labels}{tokenizer.eos_token}'
        encoded_input = tokenizer(
            template_with_labels,
            return_tensors="pt",
            add_special_tokens=False,
            return_attention_mask=True,
        )

    else:
        encoded_input = tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
    return encoded_input


def data_collate_llm_dataset(batch: list[dict], tokenizer: PreTrainedTokenizerBase) -> dict:
    """
    Collate function for DataLoader.

    :param batch: batch of data
    :param tokenizer: tokenizer
    :return: batch of data
    """
    texts = []
    labels = []
    examples = [{"input_ids": sample["input_ids"], "attention_mask": sample["attention_mask"]} for sample in batch]
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    padded = tokenizer.pad(examples, return_tensors="pt")
    for sample in batch:
        texts.append(sample["text"])
        labels.append(sample["labels"])

    return {
        "input_ids": padded["input_ids"],
        "attention_mask": padded["attention_mask"],
        "text": texts,
        "labels": labels
    }
