import pandas as pd
import numpy as np
import os


def compare_features(ideal_path, flawed_path, output_path):

    ideal = pd.read_csv(ideal_path)
    flawed = pd.read_csv(flawed_path)

    feature_columns = [
        "energy",
        "zcr",
        "spectral_centroid",
        "dominant_frequency",
        "pitch_f0"
    ]

    for i in range(1, 14):
        feature_columns.append(f"mfcc_{i}")

    common_time = np.linspace(0, 1, 500)

    comparison = pd.DataFrame()
    comparison["progress"] = common_time

    normalized_differences = []

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

        # Absolute difference
        difference = np.abs(ideal_values - flawed_values)

        # Normalize feature difference
        scale = np.std(ideal_values)

        if scale < 1e-8:
            scale = np.mean(np.abs(ideal_values)) + 1e-8

        normalized_difference = difference / scale

        comparison[f"{feature}_ideal"] = ideal_values
        comparison[f"{feature}_flawed"] = flawed_values
        comparison[f"{feature}_difference"] = normalized_difference

        normalized_differences.append(normalized_difference)

    # Overall normalized deviation
    comparison["overall_deviation"] = np.mean(
        normalized_differences,
        axis=0
    )

    output_folder = os.path.dirname(output_path)

    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    comparison.to_csv(output_path, index=False)

    print("Feature comparison successful!")
    print("Comparison points:", len(comparison))
    print("Features compared:", len(feature_columns))
    print(
        "Average normalized deviation:",
        round(comparison["overall_deviation"].mean(), 4)
    )
    print("Feature normalization applied!")
    print("Saved to:", output_path)

    return comparison


if __name__ == "__main__":

    compare_features(
        "results/ideal_001_features.csv",
        "results/flawed_001_features.csv",
        "results/feature_comparison.csv"
    )