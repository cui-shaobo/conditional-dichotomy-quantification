import pandas as pd
import os

def load_dataset(jsonl_file_path, column_mapping):
    df = pd.read_json(jsonl_file_path, lines=True)

    required_columns = list(column_mapping.keys())
    assert all(col in df.columns for col in required_columns), \
        f"Data must contain columns: {required_columns}"

    df = df.rename(columns=column_mapping)
    return df

def compute_avg_len(df, column_name):
    return df[column_name].apply(lambda x: len(x.split())).mean()

def main():
    # Define column mappings
    scenario_column_mapping = {
        'A': {
            'context_text': 'context',
            'supporter_text': 'positive',
            'defeater_text': 'negative',
            'neutral_text': 'neutral'
        },
        'B': {
            'context_text': 'context',
            'supporter_text': 'positive',
            'defeater_text': 'negative',
            'neutral_text': 'neutral'
        },
        'C': {
            'context_text': 'context',
            'supporter_text': 'positive',
            'defeater_text': 'negative',
            'neutral_text': 'neutral'
        }
    }

    # Define directory and file name mappings
    scenario_file_mapping = {
        'A': {
            'dir': 'data/perspectrum/',
            'train': 'perspectrum_train_selected_from_B_C.jsonl',
            'val': 'perspectrum_dev_processed_gpt-4o.jsonl',
            'test': 'perspectrum_test_processed_gpt-4o.jsonl'
        },
        'B': {
            'dir': 'data/defeasible_snli/',
            'train': 'defeasibleNLI_train_selected_from_A_C.jsonl',
            'val': 'defeasibleNLI_dev_processed_gpt-4o.jsonl',
            'test': 'defeasibleNLI_test_processed_gpt-4o.jsonl'
        },
        'C': {
            'dir': 'data/delta_causal/',
            'train': 'deltaCausal_train_selected_from_A_B.jsonl',
            'val': 'deltaCausal_dev_processed_gpt-4o.jsonl',
            'test': 'deltaCausal_test_processed_gpt-4o.jsonl'
        }
    }

    # Select scenario
    scenario = 'B'
    column_mapping = scenario_column_mapping[scenario]
    file_mapping = scenario_file_mapping[scenario]

    # Load datasets
    train_file_path = os.path.join(file_mapping['dir'], file_mapping['train'])
    val_file_path = os.path.join(file_mapping['dir'], file_mapping['val'])
    test_file_path = os.path.join(file_mapping['dir'], file_mapping['test'])

    train_df = load_dataset(train_file_path, column_mapping)
    val_df = load_dataset(val_file_path, column_mapping)
    test_df = load_dataset(test_file_path, column_mapping)

    # Combine datasets
    combined_df = pd.concat([train_df, val_df, test_df])

    # Statistics
    total_examples = combined_df.shape[0]
    train_examples = train_df.shape[0]
    val_examples = val_df.shape[0]
    test_examples = test_df.shape[0]

    avg_len_context = compute_avg_len(combined_df, 'context')
    avg_len_supporter = compute_avg_len(combined_df, 'positive')
    avg_len_defeater = compute_avg_len(combined_df, 'negative')
    avg_len_neutral = compute_avg_len(combined_df, 'neutral')

    # Output statistics
    print(f"Number of train examples: {train_examples}")
    print(f"Number of validation examples: {val_examples}")
    print(f"Number of test examples: {test_examples}")
    print(f"Total number of examples: {total_examples}")
    print(f"Average length of context text: {avg_len_context:.2f} tokens")
    print(f"Average length of supporter text: {avg_len_supporter:.2f} tokens")
    print(f"Average length of defeater text: {avg_len_defeater:.2f} tokens")
    print(f"Average length of neutral text: {avg_len_neutral:.2f} tokens")

if __name__ == '__main__':
    main()
