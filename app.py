import streamlit as st
import pandas as pd
import os
import tempfile

from src.feature_extraction import extract_features
from src.compare_features import compare_features
from src.flaw_detection import detect_flaws
from src.causal_explanation import generate_causal_explanations
from src.rubric_score import calculate_rubric_score


# =========================
# Page Configuration
# =========================

st.set_page_config(
    page_title="SpeechGuard AI",
    page_icon="🎙️",
    layout="wide"
)


# =========================
# Title
# =========================

st.title("🎙️ SpeechGuard AI")

st.subheader(
    "Contrastive Speech Analytics & Temporal Flaw Grounding"
)

st.write(
    "Compare an ideal speech with a flawed speech "
    "and identify temporal delivery deviations."
)


# =========================
# Upload Speech
# =========================

st.subheader("📤 Upload Your Speech")

col1, col2 = st.columns(2)

with col1:
    ideal_upload = st.file_uploader(
        "Upload Ideal Speech",
        type=["wav", "mp3", "m4a"],
        key="ideal_upload"
    )

with col2:
    flawed_upload = st.file_uploader(
        "Upload Flawed Speech",
        type=["wav", "mp3", "m4a"],
        key="flawed_upload"
    )


# =========================
# Upload Transcript
# =========================

transcript_upload = st.file_uploader(
    "Upload Transcript (TXT)",
    type=["txt"],
    key="transcript_upload"
)


# =========================
# Read Transcript
# =========================

transcript_text = ""

if transcript_upload is not None:

    st.success("✅ Transcript Uploaded")

    transcript_text = transcript_upload.read().decode(
        "utf-8"
    )

    with st.expander("📝 View Transcript"):
        st.write(transcript_text)


# =========================
# Preview Uploaded Audio
# =========================

if ideal_upload is not None:

    st.success("✅ Ideal Speech Uploaded")

    st.audio(ideal_upload)


if flawed_upload is not None:

    st.success("✅ Flawed Speech Uploaded")

    st.audio(flawed_upload)


# =========================
# Analyze Button
# =========================

if (
    ideal_upload is not None
    and flawed_upload is not None
):

    st.divider()

    if st.button(
        "🚀 Analyze Speech",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "Running SpeechGuard AI analysis..."
        ):

            try:

                # =========================
                # Create Temporary Folder
                # =========================

                temp_dir = tempfile.mkdtemp(
                    prefix="speechguard_"
                )

                ideal_extension = os.path.splitext(
                    ideal_upload.name
                )[1]

                flawed_extension = os.path.splitext(
                    flawed_upload.name
                )[1]

                ideal_path = os.path.join(
                    temp_dir,
                    "ideal_audio" + ideal_extension
                )

                flawed_path = os.path.join(
                    temp_dir,
                    "flawed_audio" + flawed_extension
                )


                # =========================
                # Save Uploaded Audio
                # =========================

                with open(
                    ideal_path,
                    "wb"
                ) as f:

                    f.write(
                        ideal_upload.getbuffer()
                    )


                with open(
                    flawed_path,
                    "wb"
                ) as f:

                    f.write(
                        flawed_upload.getbuffer()
                    )


                # =========================
                # Result Paths
                # =========================

                ideal_features = (
                    "results/uploaded_ideal_features.csv"
                )

                flawed_features = (
                    "results/uploaded_flawed_features.csv"
                )

                comparison_file = (
                    "results/feature_comparison.csv"
                )

                flaw_file = (
                    "results/detected_flaws.csv"
                )

                explanation_file = (
                    "results/causal_explanations.csv"
                )

                score_file = (
                    "results/rubric_score.csv"
                )


                # =========================
                # STEP 1
                # Feature Extraction
                # =========================

                st.write(
                    "🔹 Extracting speech features..."
                )

                extract_features(
                    ideal_path,
                    ideal_features
                )

                flawed_duration = extract_features(
                    flawed_path,
                    flawed_features
                )


                # =========================
                # STEP 2
                # Feature Comparison
                # =========================

                st.write(
                    "🔹 Comparing ideal and flawed speech..."
                )

                compare_features(
                    ideal_features,
                    flawed_features,
                    comparison_file
                )


                # =========================
                # STEP 3
                # Temporal Flaw Detection
                # =========================

                st.write(
                    "🔹 Detecting temporal flaws..."
                )

                detect_flaws(
                    comparison_file,
                    flaw_file,
                    flawed_duration
                )


                # =========================
                # STEP 4
                # Causal Explanation
                # =========================

                st.write(
                    "🔹 Generating causal explanations..."
                )

                generate_causal_explanations(
                    comparison_file,
                    flaw_file,
                    explanation_file,
                    flawed_duration
                )


                # =========================
                # STEP 5
                # Rubric Score
                # =========================

                st.write(
                    "🔹 Calculating speech quality score..."
                )

                calculate_rubric_score(
                    flaw_file,
                    score_file
                )


                st.success(
                    "🎉 Speech analysis completed successfully!"
                )

            except Exception as e:

                st.error(
                    "❌ Analysis failed."
                )

                st.exception(e)


# =========================
# Result File Paths
# =========================

comparison_file = (
    "results/feature_comparison.csv"
)

flaw_file = (
    "results/detected_flaws.csv"
)

explanation_file = (
    "results/causal_explanations.csv"
)

score_file = (
    "results/rubric_score.csv"
)


# =========================
# Load Results
# =========================

if os.path.exists(comparison_file):

    comparison = pd.read_csv(
        comparison_file
    )


    # =========================
    # Analysis Summary
    # =========================

    st.divider()

    st.subheader(
        "📊 Analysis Summary"
    )

    col1, col2, col3, col4 = st.columns(4)


    col1.metric(
        "Analysis Points",
        len(comparison)
    )


    col2.metric(
        "Average Deviation",
        round(
            comparison[
                "overall_deviation"
            ].mean(),
            2
        )
    )


    # =========================
    # Load Flaws
    # =========================

    flaws = pd.DataFrame()

    if os.path.exists(flaw_file):

        flaws = pd.read_csv(
            flaw_file
        )

        col3.metric(
            "Detected Flaw Regions",
            len(flaws)
        )

    else:

        col3.metric(
            "Detected Flaw Regions",
            0
        )


    # =========================
    # Load Quality Score
    # =========================

    score_data = pd.DataFrame()

    if os.path.exists(score_file):

        score_data = pd.read_csv(
            score_file
        )

        score = int(
            score_data[
                "speech_quality_score"
            ].iloc[0]
        )

        col4.metric(
            "Speech Quality Score",
            f"{score}/100"
        )

    else:

        score = 0

        col4.metric(
            "Speech Quality Score",
            "N/A"
        )


    # =========================
    # Quality Assessment
    # =========================

    if not score_data.empty:

        st.subheader(
            "🎯 Speech Quality Assessment"
        )

        score_col1, score_col2, score_col3, score_col4 = (
            st.columns(4)
        )


        score_col1.metric(
            "Quality Score",
            f"{score}/100"
        )


        score_col2.metric(
            "Rating",
            score_data[
                "rating"
            ].iloc[0]
        )


        score_col3.metric(
            "High Severity",
            int(
                score_data[
                    "high_severity_flaws"
                ].iloc[0]
            )
        )


        score_col4.metric(
            "Medium Severity",
            int(
                score_data[
                    "medium_severity_flaws"
                ].iloc[0]
            )
        )


    # =========================
    # Severity Distribution
    # =========================

    if not flaws.empty:

        st.divider()

        st.subheader(
            "📊 Flaw Severity Distribution"
        )

        high_count = (
            flaws["max_deviation"] > 400
        ).sum()

        medium_count = (
            (flaws["max_deviation"] > 300)
            & (flaws["max_deviation"] <= 400)
        ).sum()

        low_count = (
            flaws["max_deviation"] <= 300
        ).sum()

        severity_data = pd.DataFrame({
            "Severity": [
                "High",
                "Medium",
                "Low"
            ],
            "Count": [
                high_count,
                medium_count,
                low_count
            ]
        })

        st.bar_chart(
            severity_data.set_index(
                "Severity"
            )
        )


    # =========================
    # Temporal Deviation
    # =========================

    st.divider()

    st.subheader(
        "📈 Temporal Deviation"
    )


    if not flaws.empty:

        flawed_duration = flaws[
            "end_time"
        ].max()

        if flawed_duration <= 0:

            flawed_duration = 1.0

    else:

        flawed_duration = 1.0


    chart_data = comparison[
        [
            "progress",
            "overall_deviation"
        ]
    ].copy()


    chart_data[
        "Time (seconds)"
    ] = (
        chart_data[
            "progress"
        ] * flawed_duration
    )


    chart_data = chart_data.set_index(
        "Time (seconds)"
    )


    st.line_chart(
        chart_data[
            ["overall_deviation"]
        ]
    )


    # =========================
    # Top Critical Flaws
    # =========================

    if not flaws.empty:

        st.divider()

        st.subheader(
            "🚨 Top Critical Flaws"
        )

        critical_flaws = flaws.sort_values(
            by="max_deviation",
            ascending=False
        ).head(3)


        for _, flaw in critical_flaws.iterrows():

            if flaw["max_deviation"] > 400:

                severity = "High"
                icon = "🔴"

            elif flaw["max_deviation"] > 300:

                severity = "Medium"
                icon = "🟠"

            else:

                severity = "Low"
                icon = "🟢"


            st.warning(
                f"{icon} **Flaw "
                f"{int(flaw['region_id'])}** | "
                f"{severity} | "
                f"{flaw['start_time']:.2f}s → "
                f"{flaw['end_time']:.2f}s | "
                f"Peak deviation: "
                f"{flaw['max_deviation']:.2f}"
            )


    # =========================
    # Detected Flaw Time Ranges
    # =========================

    if not flaws.empty:

        st.subheader(
            "🔴 Detected Flaw Time Ranges"
        )


        for _, flaw in flaws.iterrows():

            if flaw["max_deviation"] > 400:

                severity = "High"

            elif flaw["max_deviation"] > 300:

                severity = "Medium"

            else:

                severity = "Low"


            st.write(
                f"⚠️ Flaw "
                f"{int(flaw['region_id'])}: "
                f"{flaw['start_time']:.2f}s → "
                f"{flaw['end_time']:.2f}s | "
                f"Duration: "
                f"{flaw['duration']:.2f}s | "
                f"Severity: {severity}"
            )


    # =========================
    # Transcript Grounding
    # =========================

    if transcript_text and not flaws.empty:

        st.divider()

        st.subheader(
            "📝 Transcript Grounding"
        )


        paragraphs = [
            p.strip()
            for p in transcript_text.split("\n\n")
            if p.strip()
        ]


        if len(paragraphs) > 0:

            for _, flaw in flaws.iterrows():

                start_time = flaw[
                    "start_time"
                ]

                end_time = flaw[
                    "end_time"
                ]


                midpoint = (
                    start_time + end_time
                ) / 2


                progress = (
                    midpoint / flawed_duration
                )


                segment_index = int(
                    progress * len(paragraphs)
                )


                segment_index = max(
                    0,
                    min(
                        segment_index,
                        len(paragraphs) - 1
                    )
                )


                matched_text = paragraphs[
                    segment_index
                ]


                st.markdown(
                    f"**Flaw "
                    f"{int(flaw['region_id'])}** "
                    f"({start_time:.2f}s → "
                    f"{end_time:.2f}s)"
                )


                st.info(
                    matched_text
                )


    # =========================
    # Detailed Flaw Analysis
    # =========================

    if os.path.exists(
        explanation_file
    ):

        explanations = pd.read_csv(
            explanation_file
        )

        st.divider()

        st.subheader(
            "⚠️ Detailed Flaw Analysis"
        )


        for _, explanation in explanations.iterrows():

            severity = explanation[
                "severity"
            ]


            # =========================
            # Severity Icon + Recommendation
            # =========================

            if severity == "High":

                icon = "🔴"

                recommendation = (
                    "Practice this portion again "
                    "with better control of speech delivery."
                )

            elif severity == "Medium":

                icon = "🟠"

                recommendation = (
                    "Focus on improving consistency "
                    "and clarity in this portion."
                )

            else:

                icon = "🟢"

                recommendation = (
                    "Minor variation detected. "
                    "Continue regular speech practice."
                )


            # =========================
            # Expandable Flaw Card
            # =========================

            with st.expander(
                f"{icon} Flaw "
                f"{int(explanation['flaw_id'])} "
                f"— {severity} Severity"
            ):


                st.write(
                    f"⏱️ **Time:** "
                    f"{explanation['start_time']:.2f}s → "
                    f"{explanation['end_time']:.2f}s"
                )


                st.write(
                    f"🎯 **Dominant Feature:** "
                    f"{explanation['dominant_feature']}"
                )


                st.write(
                    f"📊 **Feature Difference:** "
                    f"{explanation['feature_difference']:.2f}"
                )


                st.write(
                    f"📈 **Peak Deviation:** "
                    f"{explanation['peak_deviation']:.2f}"
                )


                st.write(
                    f"⚠️ **Severity:** "
                    f"{severity}"
                )


                st.info(
                    explanation["explanation"]
                )


                st.success(
                    f"💡 **Recommendation:** "
                    f"{recommendation}"
                )


    # =========================
    # Download Results
    # =========================

    st.divider()

    st.subheader(
        "📥 Download Analysis Results"
    )


    if os.path.exists(
        flaw_file
    ):

        with open(
            flaw_file,
            "rb"
        ) as f:

            st.download_button(
                "Download Detected Flaws",
                f,
                file_name="detected_flaws.csv",
                mime="text/csv"
            )


    if os.path.exists(
        explanation_file
    ):

        with open(
            explanation_file,
            "rb"
        ) as f:

            st.download_button(
                "Download Causal Explanations",
                f,
                file_name="causal_explanations.csv",
                mime="text/csv"
            )


    if os.path.exists(
        score_file
    ):

        with open(
            score_file,
            "rb"
        ) as f:

            st.download_button(
                "Download Rubric Score",
                f,
                file_name="rubric_score.csv",
                mime="text/csv"
            )


else:

    st.info(
        "Upload an Ideal Speech and a Flawed Speech, "
        "then click Analyze Speech."
    )