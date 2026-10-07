import pandas as pd
import numpy as np
import os


def detect_flaws(input_file, output_file, flawed_duration):

    df = pd.read_csv(input_file)

    deviation = df["overall_deviation"].values

    mean_deviation = np.mean(deviation)
    std_deviation = np.std(deviation)

    # Detection threshold
    threshold = mean_deviation + 1.5 * std_deviation

    is_flaw = deviation > threshold

    # =========================================================
    # FIND FLAW REGIONS
    # =========================================================

    regions = []

    start = None

    for i, value in enumerate(is_flaw):

        if value and start is None:

            start = i

        elif not value and start is not None:

            regions.append(
                (start, i - 1)
            )

            start = None

    if start is not None:

        regions.append(
            (start, len(is_flaw) - 1)
        )


    # =========================================================
    # MERGE NEARBY REGIONS
    # =========================================================

    MERGE_GAP = 10

    merged_regions = []

    for region in regions:

        if not merged_regions:

            merged_regions.append(
                list(region)
            )

            continue

        previous = merged_regions[-1]

        gap = (
            region[0]
            - previous[1]
            - 1
        )

        if gap <= MERGE_GAP:

            previous[1] = region[1]

        else:

            merged_regions.append(
                list(region)
            )


    # =========================================================
    # REMOVE VERY SMALL REGIONS
    # =========================================================

    MIN_REGION_LENGTH = 5

    final_regions = []

    for start, end in merged_regions:

        length = end - start + 1

        if length >= MIN_REGION_LENGTH:

            final_regions.append(
                (start, end)
            )


    # =========================================================
    # CREATE RESULTS
    # =========================================================

    results = []

    for region_id, (start, end) in enumerate(
        final_regions,
        start=1
    ):

        region_deviation = deviation[
            start:end + 1
        ]

        start_time = (
            df["progress"].iloc[start]
            * flawed_duration
        )

        end_time = (
            df["progress"].iloc[end]
            * flawed_duration
        )

        duration = (
            end_time - start_time
        )

        max_deviation = np.max(
            region_deviation
        )

        mean_region_deviation = np.mean(
            region_deviation
        )


        # =====================================================
        # RELATIVE SEVERITY
        # =====================================================

        # Calculate how far the region is above
        # the detection threshold.

        if threshold > 0:

            severity_ratio = (
                max_deviation / threshold
            )

        else:

            severity_ratio = 0


        if severity_ratio >= 1.50:

            severity = "High"

        elif severity_ratio >= 1.20:

            severity = "Medium"

        else:

            severity = "Low"


        results.append({

            "region_id": region_id,

            "start_index": start,

            "end_index": end,

            "start_time": start_time,

            "end_time": end_time,

            "duration": duration,

            "duration_points": (
                end - start + 1
            ),

            "max_deviation": max_deviation,

            "mean_deviation": (
                mean_region_deviation
            ),

            "severity": severity

        })


    # =========================================================
    # SAVE RESULTS
    # =========================================================

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


    # =========================================================
    # PRINT SUMMARY
    # =========================================================

    print(
        "Temporal flaw detection successful!"
    )

    print(
        f"Mean deviation: {mean_deviation:.4f}"
    )

    print(
        f"Detection threshold: {threshold:.4f}"
    )

    print(
        f"Number of flaw regions: {len(result_df)}"
    )

    if not result_df.empty:

        print(
            "\nSeverity distribution:"
        )

        print(
            result_df[
                "severity"
            ].value_counts()
        )

    print(
        f"Saved to: {output_file}"
    )

    return result_df


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    detect_flaws(
        "results/feature_comparison.csv",
        "results/detected_flaws.csv",
        16.0
    )