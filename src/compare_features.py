import pandas as pd
import numpy as np
import os


def compare_features(ideal_path, flawed_path, output_path):

    # Load feature files
    ideal = pd.read_csv(ideal_path)
    flawed = pd.read_csv(flawed_path)

    # Features to compare
    feature_columns = [
        "energy",
        "zcr",
        "spectral_centroid",
        "pitch_f0"
    ]

    for i in range(1, 14):
        feature_columns.append(f"mfcc_{i}")

    # Common normalized timeline
    common_time = np.linspace(0, 1, 500)

    comparison = pd.DataFrame()
    comparison["progress"] = common_time

    # Compare each feature
    for feature in feature_columns:

        ideal_values = np.interp(
            common_time,
            np.linspace(0, 1, len(ideal)),
            ideal[feature].values
        )

        flawed_values = np.interp(
            common_time,
            np.linspace(0, 1, len(flawed)),
            flawed[feature].values
        )

        difference = np.abs(
            ideal_values - flawed_values
        )

        comparison[
            f"{feature}_ideal"
        ] = ideal_values

        comparison[
            f"{feature}_flawed"
        ] = flawed_values

        comparison[
            f"{feature}_difference"
        ] = difference

    # Calculate overall temporal deviation
    difference_columns = [
        col for col in comparison.columns
        if col.endswith("_difference")
    ]

    comparison["overall_deviation"] = comparison[
        difference_columns
    ].mean(axis=1)

    # Save result
    output_folder = os.path.dirname(output_path)

    if output_folder:
        os.makedirs(
            output_folder,
            exist_ok=True
        )

    comparison.to_csv(
        output_path,
        index=False
    )

    print("Feature comparison successful!")
    print("Comparison points:", len(comparison))

    print(
        "Average overall deviation:",
        round(
            comparison["overall_deviation"].mean(),
            4
        )
    )

    print(
        "Saved to:",
        output_path
    )

    return comparison


# ==================================================
# TEST WITH EXISTING SAMPLE
# ==================================================

if __name__ == "__main__":

    compare_features(
        "results/ideal_001_features.csv",
        "results/flawed_001_features.csv",
        "results/feature_comparison.csv"
    )