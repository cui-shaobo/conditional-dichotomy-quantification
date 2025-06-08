"""
@Project  : dichotomous-score
@File     : plotting_embedding_visualization_main.py
@Author   : Shaobo Cui
@Date     : 01.11.2024 11:02
"""
import random

import torch
from oppositescore.model.dichotomye import DichotomyE
from visualization.embedding_visualization import EmbeddingVisualizer

custom_cache_dir = "/mnt/lia/scratch/scui/cached_files/hf"

# Define scenarios with model paths and test file paths
SCENARIOS_DOGE_BERT = {
    "A": {
        "model_path": "/mnt/lia/scratch/wenqliu/dichotomous/dichotomous-score/ckpts/Bertbase/Emsemble/A/dataset-d/seed_1015/checkpoint-1135",
        "test_file": "/mnt/lia/scratch/scui/projects/dichotomous-score/data/perspectrum/perspectrum_train_selected_from_B_C.jsonl",
        "output_prefix": "scenario_a"
    },
    "B": {
        "model_path": "/mnt/lia/scratch/wenqliu/dichotomous/dichotomous-score/ckpts/Bertbase/Emsemble/B/dataset-d/seed_1015/checkpoint-170",
        "test_file": "/mnt/lia/scratch/scui/projects/dichotomous-score/data/defeasible_snli/defeasibleNLI_train_selected_from_A_C.jsonl",
        "output_prefix": "scenario_b"
    },
    "C": {
        "model_path": "/mnt/lia/scratch/wenqliu/dichotomous/dichotomous-score/ckpts/Bertbase/Emsemble/C/dataset-d/seed_1015/checkpoint-275",
        "test_file": "/mnt/lia/scratch/scui/projects/dichotomous-score/data/delta_causal/deltaCausal_train_selected_from_A_B.jsonl",
        "output_prefix": "scenario_c"
    }
}

SCENARIOS_DOGE_ROBERTA = {
    "A": {
        "model_path": "/mnt/lia/scratch/wenqliu/dichotomous/dichotomous-score/ckpts/Robertabase/Emsemble/A/dataset-d/seed_1015/checkpoint-1135",
        "test_file": "/mnt/lia/scratch/scui/projects/dichotomous-score/data/perspectrum/perspectrum_train_selected_from_B_C.jsonl",
        "output_prefix": "scenario_a"
    },
    "B": {
        "model_path": "/mnt/lia/scratch/wenqliu/dichotomous/dichotomous-score/ckpts/Robertabase/Emsemble/B/dataset-d/seed_1015/checkpoint-170",
        "test_file": "/mnt/lia/scratch/scui/projects/dichotomous-score/data/defeasible_snli/defeasibleNLI_train_selected_from_A_C.jsonl",
        "output_prefix": "scenario_b"
    },
    "C": {
        "model_path": "/mnt/lia/scratch/wenqliu/dichotomous/dichotomous-score/ckpts/Robertabase/Emsemble/C/dataset-d/seed_1015/checkpoint-275",
        "test_file": "/mnt/lia/scratch/scui/projects/dichotomous-score/data/delta_causal/deltaCausal_train_selected_from_A_B.jsonl",
        "output_prefix": "scenario_c"
    }
}


SCENARIOS_BERTBASE = {
    "A": {
        "model_path": "google-bert/bert-base-uncased",
        "test_file": "/mnt/lia/scratch/scui/projects/dichotomous-score/data/perspectrum/perspectrum_train_selected_from_B_C.jsonl",
        "output_prefix": "scenario_a"
    },
    "B": {
        "model_path": "google-bert/bert-base-uncased",
        "test_file": "/mnt/lia/scratch/scui/projects/dichotomous-score/data/defeasible_snli/defeasibleNLI_train_selected_from_A_C.jsonl",
        "output_prefix": "scenario_b"
    },
    "C": {
        "model_path": "google-bert/bert-base-uncased",
        "test_file": "/mnt/lia/scratch/scui/projects/dichotomous-score/data/delta_causal/deltaCausal_train_selected_from_A_B.jsonl",
        "output_prefix": "scenario_c"
    }
}

SCENARIOS_ROBERTA = {
    "A": {
        "model_path": "FacebookAI/roberta-base",
        "test_file": "/mnt/lia/scratch/scui/projects/dichotomous-score/data/perspectrum/perspectrum_train_selected_from_B_C.jsonl",
        "output_prefix": "scenario_a"
    },
    "B": {
        "model_path": "FacebookAI/roberta-base",
        "test_file": "/mnt/lia/scratch/scui/projects/dichotomous-score/data/defeasible_snli/defeasibleNLI_train_selected_from_A_C.jsonl",
        "output_prefix": "scenario_b"
    },
    "C": {
        "model_path": "FacebookAI/roberta-base",
        "test_file": "/mnt/lia/scratch/scui/projects/dichotomous-score/data/delta_causal/deltaCausal_train_selected_from_A_B.jsonl",
        "output_prefix": "scenario_c"
    }
}

SCENARIOS_LLaMA3_8B = {
    "A": {
        "model_path": "meta-llama/Meta-Llama-3-8B",
        "test_file": "/mnt/lia/scratch/scui/projects/dichotomous-score/data/perspectrum/perspectrum_train_selected_from_B_C.jsonl",
        "output_prefix": "scenario_a"
    },
    "B": {
        "model_path": "meta-llama/Meta-Llama-3-8B",
        "test_file": "/mnt/lia/scratch/scui/projects/dichotomous-score/data/defeasible_snli/defeasibleNLI_train_selected_from_A_C.jsonl",
        "output_prefix": "scenario_b"
    },
    "C": {
        "model_path": "meta-llama/Meta-Llama-3-8B",
        "test_file": "/mnt/lia/scratch/scui/projects/dichotomous-score/data/delta_causal/deltaCausal_train_selected_from_A_B.jsonl",
        "output_prefix": "scenario_c"
    }
}


SCENARIOS_LLaMA3_70B = {
    "A": {
        "model_path": "meta-llama/Meta-Llama-3-70B",
        "test_file": "/mnt/lia/scratch/scui/projects/dichotomous-score/data/perspectrum/perspectrum_train_selected_from_B_C.jsonl",
        "output_prefix": "scenario_a"
    },
    "B": {
        "model_path": "meta-llama/Meta-Llama-3-70B",
        "test_file": "/mnt/lia/scratch/scui/projects/dichotomous-score/data/defeasible_snli/defeasibleNLI_train_selected_from_A_C.jsonl",
        "output_prefix": "scenario_b"
    },
    "C": {
        "model_path": "meta-llama/Meta-Llama-3-70B",
        "test_file": "/mnt/lia/scratch/scui/projects/dichotomous-score/data/delta_causal/deltaCausal_train_selected_from_A_B.jsonl",
        "output_prefix": "scenario_c"
    }
}

def main(scenario, scenarios_settings, sample_size=None, include_context=True):
    SCENARIOS = scenarios_settings
    if scenario not in SCENARIOS:
        raise ValueError(f"Scenario '{scenario}' is not defined. Available scenarios: {list(SCENARIOS.keys())}")

    # Get configuration for the selected scenario
    config = SCENARIOS[scenario]
    model_path = config["model_path"]
    test_file = config["test_file"]
    output_prefix = config["output_prefix"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        # Load model
        print(f"Loading model from: {model_path}")
        if "llama" in model_path or "Llama" in model_path:
            pooling_strategy = "last"
            model = DichotomyE.from_pretrained(model_path, cached_hf_dir=custom_cache_dir,
                                               pooling_strategy=pooling_strategy)

        else:
            pooling_strategy = "cls"
            model = DichotomyE.from_pretrained(model_path, cached_hf_dir=custom_cache_dir, pooling_strategy=pooling_strategy).to(device)

        # Load data
        print(f"Loading test data from: {test_file}")
        context, positive_samples, negative_samples, neutral_samples = EmbeddingVisualizer.load_data_from_jsonl(test_file,
                                                                                                                sample_size)

        # Modify context based on the include_context flag
        if not include_context:
            context = [""] * len(positive_samples)

        # Initialize visualizer
        visualizer = EmbeddingVisualizer(model, use_context=include_context, device=device)
        visualizer.encode_samples(context, positive_samples, negative_samples, neutral_samples)

        # Dimensionality reduction and visualization
        for method in ['tsne']:
            print(f"Applying {method.upper()} for dimensionality reduction...")
            visualizer.apply_dimensionality_reduction(method=method, n_components=2)

            # Include context setting in the save name
            context_flag = "with_context" if include_context else "no_context"
            save_name = f'{output_prefix}_{model_path.replace("/", "_")}_{context_flag}_{method}_{sample_size}.pdf'
            print(f"Saving {method.upper()} plot as: {save_name}")
            visualizer.plot(save_figure_name=save_name)
    finally:
        # Explicitly delete variables to free memory
        del model
        del visualizer
        torch.cuda.empty_cache()
        print("Memory released.")

if __name__ == "__main__":
    # Choose the scenario to run
    scenarios_settings = SCENARIOS_DOGE_ROBERTA
    for selected_scenario in ["A", "B", "C"]:
        for include_context in [True, False]:
            # selected_scenario = "A"  # Options: "A", "B", "C"
            random_seed = 42  # Set a fixed seed value for consistent results
            random.seed(random_seed)

            sample_size = 100
            # include_context = True  # Set to False to exclude context
            main(selected_scenario, scenarios_settings, sample_size, include_context)
