"""
@Project  : dichotomous-score
@File     : test_dichotomy_test_main.py
@Author   : Shaobo Cui
@Date     : 23.09.2024 16:41
"""

import argparse
import random

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from tqdm import tqdm
import os

from oppositescore.model.dichotomye import DichotomyE
from oppositescore.utils import evaluation_results_latex_format_convert

custom_cache_dir = "/mnt/lia/scratch/scui/cached_files/hf"
os.environ["TRANSFORMERS_CACHE"] = custom_cache_dir  # Set Transformers cache directory
os.environ["HF_HOME"] = custom_cache_dir  # Optional: Set Hugging Face Hub cache directory

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_dataset(jsonl_file_path):
    """
    Load dataset from the JSONL file.
    The dataset is expected to have the following columns: context_text, supporter_text, defeater_text, neutral_text
    """
    df = pd.read_json(jsonl_file_path, lines=True)

    # Ensure required columns are present
    required_columns = ['context_text', 'supporter_text', 'defeater_text', 'neutral_text']
    assert all(col in df.columns for col in required_columns), \
        f"Data must contain columns: {required_columns}"

    # Rename the columns
    df = df.rename(columns={
        'context_text': 'context',
        'supporter_text': 'positive',
        'defeater_text': 'negative',
        'neutral_text': 'neutral'
    })

    # Convert DataFrame to Hugging Face Dataset
    return Dataset.from_pandas(df)


def process_with_progress(dataset, tokenizer_fn, desc="Processing dataset"):
    """
    Manually process the dataset using the tokenizer function with tqdm to track progress.
    """
    processed_dataset = []
    for example in tqdm(dataset, desc=desc):
        processed_example = tokenizer_fn(example)
        processed_dataset.append(processed_example)
    return Dataset.from_list(processed_dataset)


def main(args):
    # Define the scenario folder paths
    scenario_dir_mapping = {
        'A': 'data/perspectrum/',
        'B': 'data/defeasible_snli/',
        'C': 'data/delta_causal/'
    }

    # Ensure the selected scenario is valid
    if args.scenario not in scenario_dir_mapping:
        raise ValueError(f"Invalid scenario: {args.scenario}. Choose from 'A', 'B', or 'C'.")

    # Construct the full file paths for the test dataset
    scenario_dir = scenario_dir_mapping[args.scenario]
    test_file_path = os.path.join(scenario_dir, 'perspectrum_dev_processed_gpt-4o.jsonl')
    # test_file_path = '/mnt/lia/scratch/wenqliu/evaluation/delta_causal/test_processed_filtered.jsonl'

    # Load the test dataset
    test_ds = load_dataset(test_file_path)

    # dichotomye = DichotomyE.from_pretrained(
    #         f'ckpts/{args.scenario}/dataset-d/',
    #     max_length=128,
    #     pooling_strategy='cls'
    # ).cuda()

    dichotomye = DichotomyE.from_pretrained(
            f'NousResearch/Llama-2-7b-chat-hf',
            cached_hf_dir=custom_cache_dir,
            max_length=128,
            pooling_strategy='last',
    ).cuda()
    # Evaluate the model
    corrcoef = dichotomye.evaluate(test_ds, batch_size=256)  # Adjust metric as necessary
    corrcoef = evaluation_results_latex_format_convert(corrcoef, output_columns=['DCF', 'DCF-positive', 'DCF-negative', 'pos_neg_degree'])
    print('Spearman\'s corrcoef:', corrcoef)


if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Load and evaluate the test dataset for different scenarios (A, B, C)")
    parser.add_argument(
        '--scenario',
        type=str,
        choices=['A', 'B', 'C'],
        default='A',  # Set scenario A as the default
        help="Select the scenario to evaluate the test dataset. Choose from 'A', 'B', or 'C'. Default is 'A'."
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )

    args = parser.parse_args()
    set_seed(args.seed)

    args = parser.parse_args()
    main(args)
