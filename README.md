# LLaMA-ABSA


## Introduction
This repository contains the code for the paper "LLaMA-Based Models for Aspect-Based Sentiment Analysis".

## Requirements
- Python 3.10
- Required packages are listed in `requirements.txt`

## Datasets
- Data is available in the `data_sota` folder.

## Running the code
- To run the code for instruction tuning, execute the following command:
```
python main.py --model microsoft/Orca-2-13b --data_path tasd/rest16 --instruction_tuning --no_wandb
```

- For few-shot scenario, execute the following command:
```
python main.py --model microsoft/Orca-2-13b --data_path tasd/rest16 --few_shot_prompt --no_wandb
```

- For zero-shot scenario, execute the following command:
```
python main.py --model microsoft/Orca-2-13b --data_path tasd/rest16 --no_wandb
```

- Model can be changed by changing the `--model` argument.
- Dataset can be changed by changing the `--data_path` argument.
- For more details, refer to the `main.py` and `src/args_utils.py` files.