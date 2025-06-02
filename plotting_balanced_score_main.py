import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the specific files for DoGE-B and DoGE-R across scenarios
scenario_files = {
    "A": {
        "DoGE-B": "A_DoGE_Bertbase_angles_data.csv",
        "DoGE-R": "A_DoGE_Robertabase_angles_data.csv"
    },
    "B": {
        "DoGE-B": "B_DoGE_Bertbase_angles_data.csv",
        "DoGE-R": "B_DoGE_Robertabase_angles_data.csv"
    },
    "C": {
        "DoGE-B": "C_DoGE_Bertbase_angles_data.csv",
        "DoGE-R": "C_DoGE_Robertabase_angles_data.csv"
    }
}

# Define column names for angles
columns = ["angles_pos_neutral", "angles_neg_neutral"]

# Define consistent colors for DoGE-B and DoGE-R
model_colors = {
    "DoGE-B": "steelblue",
    "DoGE-R": "darkorange"
}

# Create a separate plot for each scenario
for scenario, model_files in scenario_files.items():
    plt.figure(figsize=(6, 4))

    for model_name, file_name in model_files.items():
        # Load the CSV file
        file_path = os.path.join("results/angle_data", scenario, file_name)
        data = pd.read_csv(file_path)

        # Compute the difference (positive-neutral - negative-neutral)
        difference = data[columns[0]] - data[columns[1]]

        # Plot the histogram
        plt.hist(
            difference,
            bins=50,
            alpha=0.75,
            density=True,
            label=model_name,
            color=model_colors[model_name],
            edgecolor='black',  # Add edges for better distinction
            linewidth=0.5       # Thin edges for clarity
        )

    # Add labels, title, and legend
    plt.title(f"Scenario {scenario}", fontsize=14, fontweight="bold")
    plt.xlabel(r"Angle difference: $\Delta_{X, W \vert Z} - \Delta_{Y, W \vert Z}$", fontsize=12, fontweight="bold")
    plt.ylabel("Density", fontsize=12, fontweight="bold")
    plt.legend(fontsize=10, frameon=False, loc="upper left")  # Place the legend inside the plot

    # Configure ticks and spines
    plt.tick_params(axis='y', right=True, left=True, labelsize=10)
    plt.tick_params(axis='x', top=False, bottom=True, labelsize=10)

    # Save each scenario as a separate figure
    plt.tight_layout()
    plt.savefig(f"figures/score_difference_distributions/distribution_difference_DoGE_models_scenario_{scenario}.pdf")
    plt.show()
