import argparse
import functools
import logging
import os

import torch
import wandb
from peft import AutoPeftModelForCausalLM, LoraConfig
from torch.utils.data import DataLoader
from transformers import (BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
                          PreTrainedTokenizerBase)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

from src.args_utils import init_args_llm_sota
from src.data_utils import data_collate_llm_dataset
from src.llm_classifier import llm_classify
from src.llm_sota_dataset import LLMSotaDataset
from src.templates.sota_templates import INSTRUCTIONS
from src.templates.sota_templates_few_shot import FEW_SHOT_PROMPTS_SOTA


def init_logging() -> None:
    """Initialize logging."""
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_model_and_tokenizer_llm(
        model_path: str,
        load_in_8bits: bool,
        token: str,
        use_cpu: bool = False,
) -> tuple[AutoModelForCausalLM, PreTrainedTokenizerBase]:
    """Load model and tokenizer from path. Load model in 8-bit mode.

    :param model_path: path to pre-trained model or shortcut name
    :param load_in_8bits: if True, model is loaded in 8-bit mode
    :param token: token
    :param use_cpu: if True, CPU is used
    :return: model and tokenizer
    """
    bnb_config = None
    if not use_cpu:
        if load_in_8bits:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,  # load model in 8-bit precision
                low_cpu_mem_usage=True,
            )
            logging.info("Loading model in 8-bit mode")
        else:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,  # load model in 4-bit precision
                bnb_4bit_quant_type="nf4",  # pre-trained model should be quantized in 4-bit NF format
                bnb_4bit_use_double_quant=True,  # Using double quantization as mentioned in QLoRA paper
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            logging.info("Loading model in 4-bit mode")
    else:
        logging.info("Using CPU")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        token=token,
        low_cpu_mem_usage=True if not use_cpu else False,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, token=token)

    return model, tokenizer


def find_target_modules(model) -> list[str]:
    # Initialize a Set to Store Unique Layers
    unique_layers = set()

    # Iterate Over All Named Modules in the Model
    for name, module in model.named_modules():
        # Check if the Module Type Contains 'Linear4bit'
        if "Linear4bit" in str(type(module)):
            # Extract the Type of the Layer
            layer_type = name.split('.')[-1]

            # Add the Layer Type to the Set of Unique Layers
            unique_layers.add(layer_type)

    # Return the Set of Unique Layers Converted to a List
    return list(unique_layers)


def instruction_tuning(args: argparse.Namespace):
    use_cpu = True if args.use_cpu else True if not torch.cuda.is_available() else False
    device = torch.device("cpu" if use_cpu else "cuda" if torch.cuda.is_available() else "cpu")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # load model in 4-bit precision
        bnb_4bit_quant_type="nf4",  # pre-trained model should be quantized in 4-bit NF format
        bnb_4bit_use_double_quant=True,  # Using double quantization as mentioned in QLoRA paper
        bnb_4bit_compute_dtype=torch.bfloat16,  # During computation, pre-trained model should be loaded in BF16 format
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config if not use_cpu else None,
        device_map="auto" if not use_cpu else "cpu",
        use_cache=False,
        low_cpu_mem_usage=True,
        token=args.token,
    )

    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, token=args.token)
    tokenizer.padding_side = "right"

    resized_embeddings = False
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "[PAD]"
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))
        resized_embeddings = True

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        target_modules=find_target_modules(model),
        modules_to_save=None if not resized_embeddings else ["lm_head", "embed_tokens"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    if tokenizer.chat_template is None:
        if "microsoft/Orca" in args.model:
            tokenizer.chat_template = "{{ bos_token }} {% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    instruction = INSTRUCTIONS.get(args.data_path, None)
    if instruction is None:
        raise ValueError(f"Instruction not found for {args.data_path}")

    data_path_train = os.path.join("data_sota", args.data_path, "train.txt")

    train_dataset = LLMSotaDataset(
        data_path=str(data_path_train),
        tokenizer=tokenizer,
        max_data=args.max_train_data,
        instruction_tuning=True,
        instruction=instruction,
        testing=False,
    )

    data_path_dev = os.path.join("data_sota", args.data_path, "dev.txt")
    dev_dataset = LLMSotaDataset(
        data_path=str(data_path_dev),
        tokenizer=tokenizer,
        max_data=args.max_train_data,
        instruction_tuning=True,
        instruction=instruction,
        testing=False,
    )

    data_path_test = os.path.join("data_sota", args.data_path, "test.txt")
    test_dataset = LLMSotaDataset(
        data_path=str(data_path_test),
        tokenizer=tokenizer,
        max_data=args.max_train_data,
        instruction_tuning=True,
        instruction=instruction,
        testing=True,
    )

    output_dir = "output"

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=args.epochs,
        optim="paged_adamw_32bit",
        report_to=["wandb"] if not args.no_wandb else [],
        lr_scheduler_type="constant",
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        bf16=True if not use_cpu else False,
        tf32=True if not use_cpu else False,
        save_strategy="epoch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        evaluation_strategy="epoch",
        use_cpu=use_cpu,
        remove_unused_columns=True,
        load_best_model_at_end=True,
        save_total_limit=1,
        metric_for_best_model="eval_loss",
        disable_tqdm=True,
        group_by_length=True,
        dataloader_drop_last=False,
    )

    if "orca" in args.model.lower():
        response_template = tokenizer.encode("\n<|im_start|>assistant\n", add_special_tokens=False)[2:]
        assistant_text = "<|im_start|> assistant\n"
    elif "llama" in args.model.lower():
        response_template = tokenizer.encode(" [/INST]", add_special_tokens=False)[1:]
        assistant_text = "[/INST]"
    else:
        raise ValueError("Response template not defined for this model.")

    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    trainer = SFTTrainer(
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        model=model,
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
        dataset_text_field="input_ids",
        max_seq_length=1024,
        data_collator=collator,
    )

    best_model_dir = "best_model"
    logging.info("Training...")
    trainer.train()
    trainer.save_model(best_model_dir)
    logging.info("Training finished")

    if args.load_in_8bits:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # load best model
    model = AutoPeftModelForCausalLM.from_pretrained(
        best_model_dir,
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
    )

    tokenizer.padding_side = "left"
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        collate_fn=functools.partial(data_collate_llm_dataset, tokenizer=tokenizer),
        num_workers=0,
        shuffle=False,
        drop_last=False,
    )

    llm_classify(
        model=model,
        tokenizer=tokenizer,
        data_loader=test_dataloader,
        no_wandb=args.no_wandb,
        assistant_text=assistant_text,
        device=device,
    )


def main():
    init_logging()
    args = init_args_llm_sota()
    # Set system env variable HF_TOKEN to args.token
    if args.token is not None:
        os.environ["HF_TOKEN"] = args.token

    if not args.no_wandb:
        wandb.init(
            project="absa",
            entity="entity",
            config=vars(args),
            tags=[args.tag] if args.tag else [],
        )

    if args.instruction_tuning:
        instruction_tuning(args)
        wandb.finish()
        logging.info("This is the end...")
    else:
        prompting(args)


def prompting(args: argparse.Namespace):
    logging.info("Loading tokenizer and model...")
    model, tokenizer = load_model_and_tokenizer_llm(
        model_path=args.model,
        load_in_8bits=args.load_in_8bits,
        token=args.token,
        use_cpu=args.use_cpu,
    )

    logging.info("Tokenizer and model loaded")
    tokenizer.padding_side = "left"

    if tokenizer.chat_template is None:
        if "microsoft/Orca" in args.model:
            tokenizer.chat_template = "{{ bos_token }} {% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    if "orca" in args.model.lower():
        assistant_text = "<|im_start|> assistant\n"
    elif "llama" in args.model.lower():
        assistant_text = "[/INST]"
    else:
        raise ValueError("Response template not defined for this model.")

    instruction = INSTRUCTIONS.get(args.data_path, None)
    if instruction is None:
        raise ValueError(f"Instruction not found for {args.data_path}")
    if args.few_shot_prompt:
        few_shot = FEW_SHOT_PROMPTS_SOTA.get(args.data_path, None)
        if few_shot is None:
            raise ValueError(f"Few shot prompt not found for {args.data_path}")
        instruction += few_shot

    data_path_test = os.path.join("data_sota", args.data_path, "test.txt")
    test_dataset = LLMSotaDataset(
        data_path=str(data_path_test),
        tokenizer=tokenizer,
        max_data=args.max_train_data,
        instruction_tuning=True,
        instruction=instruction,
        testing=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        collate_fn=functools.partial(data_collate_llm_dataset, tokenizer=tokenizer),
        num_workers=0,
        shuffle=False,
        drop_last=False,
    )

    device = torch.device("cpu" if args.use_cpu else "cuda" if torch.cuda.is_available() else "cpu")
    llm_classify(
        model=model,
        tokenizer=tokenizer,
        data_loader=test_dataloader,
        no_wandb=args.no_wandb,
        assistant_text=assistant_text,
        device=device,
    )

    if not args.no_wandb:
        wandb.finish()

    logging.info("Finished!")


if __name__ == '__main__':
    main()
