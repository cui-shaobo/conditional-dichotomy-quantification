import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load dataset
def load_dataset(jsonl_file_path, column_mapping):
    df = pd.read_json(jsonl_file_path, lines=True)
    required_columns = list(column_mapping.keys())
    assert all(col in df.columns for col in required_columns), \
        f"Data must contain columns: {required_columns}"
    df = df.rename(columns=column_mapping)
    return df

# Function to compute the length of text
def compute_lengths(df, column_name):
    return df[column_name].apply(lambda x: len(x.split()))

# Column mappings
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

# Directory and file name mappings
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
        'test': 'test_processed_gpt-4o.jsonl'
    },
    'C': {
        'dir': 'data/delta_causal/',
        'train': 'deltaCausal_train_selected_from_A_B.jsonl',
        'val': 'deltaCausal_dev_processed_gpt-4o.jsonl',
        'test': 'deltaCausal_test_processed_gpt-4o.jsonl'
    }
}

# Collect lengths of texts for all scenarios
lengths_data = []

for scenario in ['A', 'B', 'C']:
    file_mapping = scenario_file_mapping[scenario]
    column_mapping = scenario_column_mapping[scenario]

    train_file_path = os.path.join(file_mapping['dir'], file_mapping['train'])
    val_file_path = os.path.join(file_mapping['dir'], file_mapping['val'])
    test_file_path = os.path.join(file_mapping['dir'], file_mapping['test'])

    train_df = load_dataset(train_file_path, column_mapping)
    val_df = load_dataset(val_file_path, column_mapping)
    test_df = load_dataset(test_file_path, column_mapping)

    combined_df = pd.concat([train_df, val_df, test_df])

    # Compute lengths for each column
    for column in ['context', 'positive', 'negative', 'neutral']:
        lengths = compute_lengths(combined_df, column)
        lengths_data.extend([(scenario, column, length) for length in lengths])

# Create a DataFrame for plotting
lengths_df = pd.DataFrame(lengths_data, columns=['Scenario', 'TextType', 'Length'])

# Define more vibrant colors for each text type
color_palette = {
    'context': '#4e79a7',  # Soft blue
    'positive': '#59a14f',  # Soft green
    'negative': '#e15759',  # Soft red
    'neutral': '#9c755f'   # Soft brown
}

# Plot and save figures for each scenario
scenarios = ['A', 'B', 'C']
for scenario in scenarios:
    scenario_data = lengths_df[lengths_df['Scenario'] == scenario]

    # Create figure for the specific scenario
    fig, ax = plt.subplots(figsize=(5, 3.1))

    sns.violinplot(
        ax=ax,
        x='TextType',
        y='Length',
        data=scenario_data,
        palette=[color_palette[text_type] for text_type in scenario_data['TextType'].unique()]
    )

    # Set labels and title adjustments
    ax.set_xlabel('')  # Remove x-axis label
    if scenario == 'A':
        ax.set_ylabel('Length distributions', fontsize=16)
    else:
        ax.set_ylabel('')  # Remove y-axis label for others

    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Adjust layout
    plt.tight_layout()

    # Save each scenario's figure as a PDF
    plt.savefig(f'sentence_length_violin_plot_{scenario}.pdf', format='pdf')

    plt.close()  # Close the figure to avoid overlapping plots

print("Figures saved for each scenario.")
