"""
@Project  : dichotomous-score
@File     : training_dichotomybert_train_main.py
@Author   : Shaobo Cui
@Date     : 08.09.2024 19:01
"""

import os
import argparse
import pandas as pd
from datasets import Dataset
from tqdm import tqdm
import random
import numpy as np
import torch
from transformers import AutoModel
from oppositescore.trainer.dichotomy_dataset import DichotomyEDataTokenizer
from oppositescore.model.dichotomye import DichotomyE

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
    Load dataset from JSONL file and rename columns.
    """
    df = pd.read_json(jsonl_file_path, lines=True)

    # Ensure required columns are present
    required_columns = ['context_text', 'supporter_text', 'defeater_text', 'neutral_text']
    assert all(col in df.columns for col in required_columns), \
        f"Data must contain columns: {required_columns}"

    # Rename columns for consistency
    df = df.rename(columns={
        'context_text': 'context',
        'supporter_text': 'positive',
        'defeater_text': 'negative',
        'neutral_text': 'neutral'
    })

    return Dataset.from_pandas(df)


def process_with_progress(dataset, tokenizer_fn, desc="Processing dataset"):
    """
    Process the dataset with a tokenizer and show progress.
    """
    processed_dataset = []
    for example in tqdm(dataset, desc=desc):
        processed_example = tokenizer_fn(example)
        processed_dataset.append(processed_example)
    return Dataset.from_list(processed_dataset)


def main(args):
    # Define scenario paths
    scenario_dir_mapping = {
        'A': 'data/perspectrum/',
        'B': 'data/defeasible_snli/',
        'C': 'data/delta_causal/'
    }

    # Ensure valid scenario
    if args.scenario not in scenario_dir_mapping:
        raise ValueError(f"Invalid scenario: {args.scenario}. Choose from 'A', 'B', or 'C'.")

    scenario_dir = scenario_dir_mapping[args.scenario]
    train_file_path = os.path.join(scenario_dir, 'deltaCausal_train_selected_from_A_B.jsonl')
    test_file_path = os.path.join(scenario_dir, 'perspectrum_dev_processed_gpt-4o.jsonl')

    # Load data
    train_ds = load_dataset(train_file_path).shuffle()
    test_ds = load_dataset(test_file_path)

    # Load model with device_map="auto"
    print("Loading model with device_map='auto'...")
    dichotomye = DichotomyE.from_pretrained(
        'google-bert/bert-base-uncased',
        max_length=128,
        pooling_strategy='cls',
        cached_hf_dir=custom_cache_dir,
        device_map='auto'
    )


    # Tokenize data
    tokenizer_fn = DichotomyEDataTokenizer(dichotomye.tokenizer, dichotomye.max_length)
    train_ds = process_with_progress(train_ds, tokenizer_fn, desc="Tokenizing train dataset")

    # Fit the model
    batch_size = 8
    print('#'* 100 + '\n' + str(len(train_ds)))
    if len(train_ds) == 0:
        raise ValueError("Training dataset is empty. Check the input dataset.")

    dichotomye.fit(
        train_ds=train_ds,
        valid_ds=None,
        output_dir=f'ckpts/{args.scenario}/dataset-d/seed_{str(args.seed)}',
        batch_size=batch_size,
        epochs=3,
        learning_rate=2e-5,
        save_steps=2000,
        eval_steps=200,
        warmup_steps=0,
        gradient_accumulation_steps=4,
        loss_kwargs={
            'cosine_w': 1.0,
            'ibn_w': 20.0,
            'angle_w': 1.0,
            'dichotomy_w': 0.0,
            'dichotomy_contrastive_w': 1.0,
            'cosine_tau': 20,
            'ibn_tau': 20,
            'angle_tau': 20,
            'dichotomy_tau': 20,
            'dichotomy_contrastive_tau': 1.0
        },
        fp16=True,
        logging_steps=100
    )

    # Evaluate the model
    corrcoef = dichotomye.evaluate(test_ds)
    print('Spearman\'s corrcoef:', corrcoef)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load and process dataset for different scenarios (A, B, C)")
    parser.add_argument(
        '--scenario',
        type=str,
        choices=['A', 'B', 'C'],
        default='C',
        help="Select the scenario to load the dataset. Choose from 'A', 'B', or 'C'."
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )

    args = parser.parse_args()
    set_seed(args.seed)
    main(args)