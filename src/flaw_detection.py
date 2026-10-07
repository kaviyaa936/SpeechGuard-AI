import pandas as pd
import numpy as np
import os


def detect_flaws(input_file, output_file, flawed_duration):

    df = pd.read_csv(input_file)

    deviation = df["overall_deviation"].values

    mean_deviation = np.mean(deviation)
    std_deviation = np.std(deviation)

    threshold = mean_deviation + 1.5 * std_deviation

    is_flaw = deviation > threshold

    regions = []
    start = None

    for i, value in enumerate(is_flaw):

        if value and start is None:
            start = i

        elif not value and start is not None:
            regions.append((start, i - 1))
            start = None

    if start is not None:
        regions.append(
            (start, len(is_flaw) - 1)
        )

    # -----------------------------
    # Merge nearby flaw regions
    # -----------------------------

    MERGE_GAP = 10

    merged_regions = []

    for region in regions:

        if not merged_regions:
            merged_regions.append(list(region))
            continue

        previous = merged_regions[-1]

        gap = region[0] - previous[1] - 1

        if gap <= MERGE_GAP:
            previous[1] = region[1]

        else:
            merged_regions.append(
                list(region)
            )

    # -----------------------------
    # Remove very tiny regions
    # -----------------------------

    MIN_REGION_LENGTH = 5

    final_regions = []

    for start, end in merged_regions:

        length = end - start + 1

        if length >= MIN_REGION_LENGTH:
            final_regions.append(
                (start, end)
            )

    # -----------------------------
    # Create flaw results
    # -----------------------------

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

        duration = end_time - start_time

        results.append({

            "region_id": region_id,

            "start_index": start,

            "end_index": end,

            "start_time": start_time,

            "end_time": end_time,

            "duration": duration,

            "duration_points": end - start + 1,

            "max_deviation": np.max(
                region_deviation
            ),

            "mean_deviation": np.mean(
                region_deviation
            )
        })

    result_df = pd.DataFrame(results)

    # -----------------------------
    # Save result
    # -----------------------------

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

    # -----------------------------
    # Print summary
    # -----------------------------

    print(
        "Temporal flaw detection completed!"
    )

    print(
        f"Flawed speech duration: "
        f"{flawed_duration:.2f} seconds"
    )

    print(
        f"Mean deviation: "
        f"{mean_deviation:.4f}"
    )

    print(
        f"Detection threshold: "
        f"{threshold:.4f}"
    )

    print(
        f"Initial flaw regions: "
        f"{len(regions)}"
    )

    print(
        f"After merging: "
        f"{len(merged_regions)}"
    )

    print(
        f"Final meaningful flaw regions: "
        f"{len(final_regions)}"
    )

    print(
        f"Saved to: "
        f"{output_file}"
    )

    return result_df


# ==================================================
# TEST WITH EXISTING SAMPLE
# ==================================================

if __name__ == "__main__":

    detect_flaws(
        "results/feature_comparison.csv",
        "results/detected_flaws.csv",
        16.0
    )