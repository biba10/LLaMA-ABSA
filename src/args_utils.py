import argparse
import logging


def init_args_llm_sota() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        choices=["acos/rest16", "acos/laptop16", "asqp/rest15", "asqp/rest16", "aste/laptop14", "aste/rest14",
                 "aste/rest15", "aste/rest16", "tasd/rest15", "tasd/rest16"],
        help="Path to data.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/Orca-2-13b",
        help="Path to pre-trained model or shortcut name.",
    )

    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Token for loading model.",
    )

    parser.add_argument(
        "--load_in_8bits",
        default=False,
        action="store_true",
        help="Use 8-bit precision.",
    )

    parser.add_argument(
        "--max_test_data",
        default=0,
        type=int,
        help="Amount of data that will be used for testing",
    )

    parser.add_argument(
        "--max_train_data",
        default=0,
        type=int,
        help="Amount of data that will be used for training",
    )

    parser.add_argument(
        "--max_dev_data",
        default=0,
        type=int,
        help="Amount of data that will be used for validation",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--no_wandb",
        default=False,
        action="store_true",
        help="Do not use WandB.",
    )

    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Tag for WandB.",
    )

    parser.add_argument(
        "--use_cpu",
        default=False,
        action="store_true",
        help="Use CPU even if GPU is available.",
    )

    group = parser.add_mutually_exclusive_group(required=False)

    group.add_argument(
        "--few_shot_prompt",
        default=False,
        action="store_true",
        help="Use few shot.",
    )

    group.add_argument(
        "--instruction_tuning",
        default=False,
        action="store_true",
        help="Use instruction tuning.",
    )

    args = parser.parse_args()

    logging.info("Arguments: %s", args)

    return args
