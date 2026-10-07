import pandas as pd
import os


def calculate_rubric_score(
    flaws_file,
    output_file
):

    # Load detected flaws
    flaws = pd.read_csv(
        flaws_file
    )

    # Count severity levels
    high_count = (
        flaws["max_deviation"] > 400
    ).sum()

    medium_count = (
        (flaws["max_deviation"] > 300) &
        (flaws["max_deviation"] <= 400)
    ).sum()

    low_count = (
        flaws["max_deviation"] <= 300
    ).sum()

    # Total number of flaws
    total_flaws = len(flaws)

    # Calculate penalty
    penalty = (
        high_count * 8 +
        medium_count * 5 +
        low_count * 2
    )

    # Calculate speech quality score
    score = max(
        0,
        100 - penalty
    )

    # Overall rating
    if score >= 85:
        rating = "Excellent"

    elif score >= 70:
        rating = "Good"

    elif score >= 50:
        rating = "Needs Improvement"

    else:
        rating = "Poor"

    # Create result
    result = pd.DataFrame([{

        "speech_quality_score": score,

        "rating": rating,

        "total_flaws": total_flaws,

        "high_severity_flaws": high_count,

        "medium_severity_flaws": medium_count,

        "low_severity_flaws": low_count
    }])

    # Create output folder
    output_folder = os.path.dirname(
        output_file
    )

    if output_folder:
        os.makedirs(
            output_folder,
            exist_ok=True
        )

    # Save result
    result.to_csv(
        output_file,
        index=False
    )

    print(
        "Rubric score generation successful!"
    )

    print(
        f"Speech Quality Score: {score}/100"
    )

    print(
        f"Rating: {rating}"
    )

    print(
        f"Total flaws: {total_flaws}"
    )

    print(
        f"High severity: {high_count}"
    )

    print(
        f"Medium severity: {medium_count}"
    )

    print(
        f"Low severity: {low_count}"
    )

    print(
        f"Saved to: {output_file}"
    )

    return result


# ==================================================
# TEST WITH EXISTING SAMPLE
# ==================================================

if __name__ == "__main__":

    calculate_rubric_score(
        "results/detected_flaws.csv",
        "results/rubric_score.csv"
    )