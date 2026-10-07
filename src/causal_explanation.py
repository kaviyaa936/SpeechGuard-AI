import pandas as pd
import os


def generate_causal_explanations(
    features_file,
    flaws_file,
    output_file,
    flawed_duration
):

    # Load feature comparison data
    features = pd.read_csv(features_file)

    # Load detected flaw regions
    flaws = pd.read_csv(flaws_file)

    explanations = []

    # Features used for causal analysis
    feature_columns = {
        "Energy": "energy_difference",
        "Zero Crossing Rate": "zcr_difference",
        "Spectral Centroid": "spectral_centroid_difference",
        "Pitch": "pitch_f0_difference",
        "MFCC 1": "mfcc_1_difference",
        "MFCC 2": "mfcc_2_difference",
        "MFCC 3": "mfcc_3_difference",
        "MFCC 4": "mfcc_4_difference",
        "MFCC 5": "mfcc_5_difference",
        "MFCC 6": "mfcc_6_difference",
        "MFCC 7": "mfcc_7_difference",
        "MFCC 8": "mfcc_8_difference",
        "MFCC 9": "mfcc_9_difference",
        "MFCC 10": "mfcc_10_difference",
        "MFCC 11": "mfcc_11_difference",
        "MFCC 12": "mfcc_12_difference",
        "MFCC 13": "mfcc_13_difference"
    }

    for index, row in flaws.iterrows():

        start_time = row["start_time"]
        end_time = row["end_time"]
        peak = row["max_deviation"]

        # Convert time range into progress range
        start_progress = start_time / flawed_duration
        end_progress = end_time / flawed_duration

        # Find corresponding feature rows
        region = features[
            (features["progress"] >= start_progress) &
            (features["progress"] <= end_progress)
        ]

        # Find dominant feature
        feature_scores = {}

        for feature_name, column_name in feature_columns.items():

            if len(region) > 0:

                feature_scores[feature_name] = (
                    region[column_name]
                    .abs()
                    .mean()
                )

            else:

                feature_scores[feature_name] = 0

        dominant_feature = max(
            feature_scores,
            key=feature_scores.get
        )

        dominant_value = feature_scores[
            dominant_feature
        ]

        # Determine severity
        if peak > 400:
            severity = "High"

        elif peak > 300:
            severity = "Medium"

        else:
            severity = "Low"

        # Feature-specific explanation
        if dominant_feature == "Energy":

            cause = (
                "The speech energy level changed significantly "
                "compared with the ideal speech."
            )

        elif dominant_feature == "Zero Crossing Rate":

            cause = (
                "The Zero Crossing Rate changed significantly, "
                "indicating a change in speech sound characteristics."
            )

        elif dominant_feature == "Spectral Centroid":

            cause = (
                "The spectral characteristics of the speech changed "
                "significantly compared with the ideal speech."
            )

        elif dominant_feature == "Pitch":

            cause = (
                "The speaker's pitch pattern changed significantly "
                "compared with the ideal speech."
            )

        else:

            cause = (
                f"{dominant_feature} changed significantly, "
                "indicating a difference in the acoustic speech pattern."
            )

        explanation = (
            f"Speech flaw detected from "
            f"{start_time:.2f} seconds to "
            f"{end_time:.2f} seconds. "

            f"The dominant acoustic change was "
            f"{dominant_feature} with an average difference of "
            f"{dominant_value:.2f}. "

            f"{cause} "

            f"Peak deviation: {peak:.2f}. "

            f"Severity: {severity}."
        )

        explanations.append({

            "flaw_id": index + 1,

            "start_time": start_time,

            "end_time": end_time,

            "dominant_feature": dominant_feature,

            "feature_difference": dominant_value,

            "peak_deviation": peak,

            "severity": severity,

            "explanation": explanation
        })

    # Create result table
    result = pd.DataFrame(
        explanations
    )

    # Ensure results folder exists
    output_folder = os.path.dirname(
        output_file
    )

    if output_folder:
        os.makedirs(
            output_folder,
            exist_ok=True
        )

    # Save causal explanations
    result.to_csv(
        output_file,
        index=False
    )

    print(
        "Feature-specific causal explanation completed!"
    )

    print(
        "Number of explanations:",
        len(result)
    )

    print(
        "Saved to:",
        output_file
    )

    return result


# ==================================================
# TEST WITH EXISTING SAMPLE
# ==================================================

if __name__ == "__main__":

    generate_causal_explanations(
        "results/feature_comparison.csv",
        "results/detected_flaws.csv",
        "results/causal_explanations.csv",
        16.0
    )