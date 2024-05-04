import ast
import logging

import torch
import wandb
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase

from src.f1_score_seq2seq import F1ScoreSeq2Seq


def _extract_prediction_from_output(outputs: list[str], assistant_text: str) -> list[list[str | tuple]]:
    """
    Extract prediction from the output of the model.

    :param outputs: list of outputs
    :param assistant_text: assistant text
    :return: list of predictions
    """
    predictions = []
    for output in outputs:
        logging.info("Output: %s", output)
        try:
            prediction = output.split(assistant_text)[-1].strip()
            if "Sentiment elements:" in prediction:
                prediction = prediction.split("Sentiment elements:")[-1].strip()
            if "utput:" in prediction:
                prediction = prediction.split("utput:")[-1].strip()
            # If there are another lines, we need to remove them
            prediction = prediction.split("\n")[0].strip()
            # convert the string into python list, it should be already in a Python list format
            logging.info("Trying to decode prediction: %s", prediction)
            prediction = ast.literal_eval(prediction)
            if not prediction or not isinstance(prediction, list):
                logging.error("Could not decode prediction: %s", prediction)
                prediction = []
            predictions.append(prediction)
        except Exception as e:
            logging.error("Could not extract prediction from output: %s", e)
            predictions.append([])

    return predictions


def llm_classify(
        model: AutoModelForCausalLM,
        tokenizer: PreTrainedTokenizerBase,
        data_loader: DataLoader,
        no_wandb: bool,
        assistant_text: str,
        device: torch.device,
) -> None:
    """
    Classify the data using the model.

    :param model: model
    :param tokenizer: tokenizer
    :param data_loader: data loader
    :param no_wandb: if True, wandb is not used
    :param assistant_text: assistant text
    :param device: device
    :return: None
    """
    model.eval()
    f1_score = F1ScoreSeq2Seq()

    for data in data_loader:
        with torch.no_grad():
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            labels = data["labels"]
            texts = data["text"]

            output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask)
            output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            logging.info("Output batch: %s", output)

            predictions = _extract_prediction_from_output(output, assistant_text)

            for prediction, label in zip(predictions, labels):
                f1_score.update(prediction, label)

            for text, prediction, label in zip(texts, predictions, labels):
                logging.info("Text: %s", text)
                logging.info("Prediction: %s", prediction)
                logging.info("Label: %s", label)

            logging.info("F1: %f", f1_score.compute())

            if not no_wandb:
                try:
                    wandb.log({"f1": f1_score.compute()})
                except Exception as e:
                    logging.error("Could not log to wandb: %s", e)
