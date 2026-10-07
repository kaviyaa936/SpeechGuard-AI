import pandas as pd
import numpy as np
import os


def generate_causal_explanations(
    flaw_file,
    comparison_file,
    output_file,
    flawed_duration
):

    flaws = pd.read_csv(
        flaw_file
    )

    comparison = pd.read_csv(
        comparison_file
    )

    # =================================================
    # FEATURES
    # =================================================

    feature_columns = [
        "energy",
        "zcr",
        "spectral_centroid",
        "dominant_frequency",
        "pitch_f0"
    ]

    for i in range(1, 14):

        feature_columns.append(
            f"mfcc_{i}"
        )

    # =================================================
    # FEATURE DESCRIPTIONS
    # =================================================

    feature_causes = {

        "energy":
            "The energy changed significantly, indicating a change in speech intensity or loudness.",

        "zcr":
            "The zero-crossing rate changed significantly, indicating a change in signal activity or articulation characteristics.",

        "spectral_centroid":
            "The spectral centroid changed significantly, indicating a change in the frequency distribution of the speech signal.",

        "dominant_frequency":
            "The dominant frequency changed significantly, indicating a change in the strongest frequency component of the speech signal.",

        "pitch_f0":
            "The pitch changed significantly, indicating a variation in the perceived highness or lowness of the voice."
    }

    feature_feedback = {

        "energy":
            "Maintain a more consistent speaking intensity.",

        "zcr":
            "Maintain clearer and more stable articulation.",

        "spectral_centroid":
            "Maintain a more consistent vocal tone and articulation.",

        "dominant_frequency":
            "Maintain a more stable frequency pattern and vocal delivery.",

        "pitch_f0":
            "Maintain a more consistent pitch and avoid sudden pitch changes."
    }

    # =================================================
    # RESULT LIST
    # =================================================

    results = []

    # =================================================
    # PROCESS EACH FLAW
    # =================================================

    for _, flaw in flaws.iterrows():

        flaw_id = flaw["region_id"]

        start_time = float(
            flaw["start_time"]
        )

        end_time = float(
            flaw["end_time"]
        )

        severity = flaw[
            "severity"
        ]

        # ---------------------------------------------
        # Convert actual time to normalized progress
        # ---------------------------------------------

        start_progress = (
            start_time
            / flawed_duration
        )

        end_progress = (
            end_time
            / flawed_duration
        )

        # ---------------------------------------------
        # Select comparison region
        # ---------------------------------------------

        region = comparison[
            (
                comparison["progress"]
                >= start_progress
            )
            &
            (
                comparison["progress"]
                <= end_progress
            )
        ]

        # Fallback if region is empty
        if region.empty:

            region = comparison.iloc[
                [
                    (
                        comparison[
                            "progress"
                        ]
                        - start_progress
                    ).abs().idxmin()
                ]
            ]

        # =================================================
        # FEATURE DIFFERENCE
        # =================================================

        feature_scores = {}

        for feature in feature_columns:

            column = (
                f"{feature}_difference"
            )

            if column in region.columns:

                value = np.mean(
                    np.abs(
                        region[column]
                    )
                )

                feature_scores[
                    feature
                ] = value

        # =================================================
        # DOMINANT FEATURE
        # =================================================

        if feature_scores:

            dominant_feature = max(
                feature_scores,
                key=feature_scores.get
            )

            dominant_value = (
                feature_scores[
                    dominant_feature
                ]
            )

        else:

            dominant_feature = "Unknown"

            dominant_value = 0.0

        # =================================================
        # PEAK DEVIATION
        # =================================================

        peak = region[
            "overall_deviation"
        ].max()

        # =================================================
        # FEATURE CAUSE
        # =================================================

        cause = feature_causes.get(
            dominant_feature,
            "The dominant acoustic feature changed significantly from the ideal reference."
        )

        feedback = feature_feedback.get(
            dominant_feature,
            "Try to maintain a more consistent speech pattern."
        )

        # =================================================
        # MATHEMATICAL RATIONALE
        # =================================================

        mathematical_rationale = (

            "The system compares the absolute feature "
            "difference between the ideal and flawed "
            "speech over this time region. "

            f"The average difference for "
            f"{dominant_feature} was "
            f"{dominant_value:.2f}, while the peak "
            f"overall deviation was "
            f"{peak:.2f}. "

            "A larger deviation indicates a stronger "
            "departure from the ideal reference."
        )

        # =================================================
        # EXPLANATION
        # =================================================

        explanation = (

            f"Speech flaw detected from "
            f"{start_time:.2f}s to "
            f"{end_time:.2f}s. "

            f"Dominant acoustic feature: "
            f"{dominant_feature}. "

            f"Average feature difference: "
            f"{dominant_value:.2f}. "

            f"{cause} "

            f"{mathematical_rationale} "

            f"Severity: {severity}. "

            f"Actionable feedback: "
            f"{feedback}"
        )

        # =================================================
        # STORE RESULT
        # =================================================

        results.append({

            "flaw_id":
                flaw_id,

            "start_time":
                start_time,

            "end_time":
                end_time,

            "dominant_feature":
                dominant_feature,

            "feature_difference":
                dominant_value,

            "peak_deviation":
                peak,

            "severity":
                severity,

            "mathematical_rationale":
                mathematical_rationale,

            "feedback":
                feedback,

            "explanation":
                explanation
        })

    # =================================================
    # SAVE RESULTS
    # =================================================

    result_df = pd.DataFrame(
        results
    )

    output_folder = os.path.dirname(
        output_file
    )

    if output_folder:

        os.makedirs(
            output_folder,
            exist_ok=True
        )

    result_df.to_csv(
        output_file,
        index=False
    )

    print(
        "Causal explanation generation successful!"
    )

    print(
        "Number of explanations:",
        len(result_df)
    )

    print(
        "Dominant frequency analysis included!"
    )

    print(
        "Saved to:",
        output_file
    )

    return result_df


# ==================================================
# TEST
# ==================================================

if __name__ == "__main__":

    generate_causal_explanations(

        "results/detected_flaws.csv",

        "results/feature_comparison.csv",

        "results/causal_explanations.csv",

        16.0
    )