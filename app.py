import streamlit as st
import pandas as pd
import os
import tempfile

from src.feature_extraction import extract_features
from src.compare_features import compare_features
from src.flaw_detection import detect_flaws
from src.causal_explanation import generate_causal_explanations
from src.rubric_score import calculate_rubric_score


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="SpeechGuard AI",
    page_icon="🎙️",
    layout="wide"
)


# =========================================================
# SESSION STATE
# =========================================================

if "analysis_completed" not in st.session_state:
    st.session_state.analysis_completed = False


# =========================================================
# TITLE
# =========================================================

st.title("🎙️ SpeechGuard AI")

st.subheader(
    "Contrastive Speech Analytics & Temporal Flaw Grounding"
)

st.write(
    "Compare an ideal speech with a flawed speech "
    "and identify temporal delivery deviations."
)


# =========================================================
# SPEECH INPUT
# =========================================================

st.subheader("🎙️ Speech Input")

st.info(
    "You can either upload an audio file or record speech directly "
    "using your microphone."
)

col1, col2 = st.columns(2)


# =========================================================
# IDEAL SPEECH
# =========================================================

with col1:

    st.markdown("### 🟢 Ideal Speech")

    ideal_record = st.audio_input(
        "🎙️ Record Ideal Speech",
        key="ideal_record"
    )

    st.caption("OR upload an existing recording")

    ideal_upload = st.file_uploader(
        "Upload Ideal Speech",
        type=["wav", "mp3", "m4a"],
        key="ideal_upload"
    )


# =========================================================
# FLAWED SPEECH
# =========================================================

with col2:

    st.markdown("### 🔴 Flawed Speech")

    flawed_record = st.audio_input(
        "🎙️ Record Flawed Speech",
        key="flawed_record"
    )

    st.caption("OR upload an existing recording")

    flawed_upload = st.file_uploader(
        "Upload Flawed Speech",
        type=["wav", "mp3", "m4a"],
        key="flawed_upload"
    )


# =========================================================
# SELECT AUDIO SOURCE
# =========================================================

ideal_audio = (
    ideal_record
    if ideal_record is not None
    else ideal_upload
)

flawed_audio = (
    flawed_record
    if flawed_record is not None
    else flawed_upload
)


# =========================================================
# SHOW AUDIO STATUS
# =========================================================

if ideal_record is not None:

    st.success(
        "🎙️ Ideal Speech recorded successfully!"
    )

elif ideal_upload is not None:

    st.success(
        "📁 Ideal Speech uploaded successfully!"
    )


if flawed_record is not None:

    st.success(
        "🎙️ Flawed Speech recorded successfully!"
    )

elif flawed_upload is not None:

    st.success(
        "📁 Flawed Speech uploaded successfully!"
    )


# =========================================================
# AUDIO PREVIEW
# =========================================================

if ideal_audio is not None:

    st.markdown(
        "#### 🟢 Ideal Speech Preview"
    )

    st.audio(ideal_audio)


if flawed_audio is not None:

    st.markdown(
        "#### 🔴 Flawed Speech Preview"
    )

    st.audio(flawed_audio)


# =========================================================
# TRANSCRIPT
# =========================================================

st.divider()

st.subheader("📝 Transcript")

transcript_upload = st.file_uploader(
    "Upload Transcript (TXT) - Optional",
    type=["txt"],
    key="transcript_upload"
)

transcript_text = ""

if transcript_upload is not None:

    st.success(
        "✅ Transcript Uploaded"
    )

    transcript_text = transcript_upload.read().decode(
        "utf-8"
    )

    with st.expander("📝 View Transcript"):

        st.write(transcript_text)


# =========================================================
# ANALYZE BUTTON
# =========================================================

if (
    ideal_audio is not None
    and flawed_audio is not None
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

                st.session_state.analysis_completed = False


                # =================================================
                # RESULT FILES
                # =================================================

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


                # =================================================
                # REMOVE OLD RESULTS
                # =================================================

                old_result_files = [

                    ideal_features,
                    flawed_features,
                    comparison_file,
                    flaw_file,
                    explanation_file,
                    score_file

                ]

                for file_path in old_result_files:

                    if os.path.exists(file_path):

                        os.remove(file_path)


                # =================================================
                # TEMPORARY FOLDER
                # =================================================

                temp_dir = tempfile.mkdtemp(
                    prefix="speechguard_"
                )


                # =================================================
                # AUDIO FILE PATHS
                # =================================================

                ideal_extension = os.path.splitext(
                    ideal_audio.name
                )[1]

                flawed_extension = os.path.splitext(
                    flawed_audio.name
                )[1]


                if not ideal_extension:

                    ideal_extension = ".wav"

                if not flawed_extension:

                    flawed_extension = ".wav"


                ideal_path = os.path.join(
                    temp_dir,
                    "ideal_audio" + ideal_extension
                )

                flawed_path = os.path.join(
                    temp_dir,
                    "flawed_audio" + flawed_extension
                )


                # =================================================
                # SAVE IDEAL AUDIO
                # =================================================

                with open(
                    ideal_path,
                    "wb"
                ) as f:

                    f.write(
                        ideal_audio.getbuffer()
                    )


                # =================================================
                # SAVE FLAWED AUDIO
                # =================================================

                with open(
                    flawed_path,
                    "wb"
                ) as f:

                    f.write(
                        flawed_audio.getbuffer()
                    )


                # =================================================
                # STEP 1 - FEATURE EXTRACTION
                # =================================================

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


                # =================================================
                # STEP 2 - FEATURE COMPARISON
                # =================================================

                st.write(
                    "🔹 Comparing ideal and flawed speech..."
                )

                compare_features(
                    ideal_features,
                    flawed_features,
                    comparison_file
                )


                # =================================================
                # STEP 3 - TEMPORAL FLAW DETECTION
                # =================================================

                st.write(
                    "🔹 Detecting temporal flaws..."
                )

                detect_flaws(
                    comparison_file,
                    flaw_file,
                    flawed_duration
                )


                # =================================================
                # STEP 4 - CAUSAL EXPLANATION
                # =================================================

                st.write(
                    "🔹 Generating causal explanations..."
                )

                # IMPORTANT:
                # 1st argument = detected flaws
                # 2nd argument = feature comparison

                generate_causal_explanations(
                    flaw_file,
                    comparison_file,
                    explanation_file,
                    flawed_duration
                )


                # =================================================
                # STEP 5 - RUBRIC SCORE
                # =================================================

                st.write(
                    "🔹 Calculating speech quality score..."
                )

                calculate_rubric_score(
                    flaw_file,
                    score_file
                )


                # =================================================
                # ANALYSIS COMPLETE
                # =================================================

                st.session_state.analysis_completed = True

                st.success(
                    "🎉 Speech analysis completed successfully!"
                )

                st.rerun()


            except Exception as e:

                st.session_state.analysis_completed = False

                st.error(
                    "❌ Analysis failed."
                )

                st.exception(e)


# =========================================================
# RESULT FILE PATHS
# =========================================================

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


# =========================================================
# SHOW RESULTS
# =========================================================

if st.session_state.analysis_completed:

    if os.path.exists(comparison_file):

        comparison = pd.read_csv(
            comparison_file
        )


        # =================================================
        # ANALYSIS SUMMARY
        # =================================================

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


        # =================================================
        # LOAD FLAWS
        # =================================================

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


        # =================================================
        # LOAD QUALITY SCORE
        # =================================================

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


        # =================================================
        # QUALITY ASSESSMENT
        # =================================================

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


        # =================================================
        # SEVERITY DISTRIBUTION
        # =================================================

        if not flaws.empty:

            st.divider()

            st.subheader(
                "📊 Flaw Severity Distribution"
            )


            high_count = (
                flaws["severity"] == "High"
            ).sum()


            medium_count = (
                flaws["severity"] == "Medium"
            ).sum()


            low_count = (
                flaws["severity"] == "Low"
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


        # =================================================
        # TEMPORAL DEVIATION
        # =================================================

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


        # =================================================
        # TOP CRITICAL FLAWS
        # =================================================

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

                severity = flaw["severity"]


                if severity == "High":

                    icon = "🔴"

                elif severity == "Medium":

                    icon = "🟠"

                else:

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


        # =================================================
        # DETECTED FLAW TIME RANGES
        # =================================================

        if not flaws.empty:

            st.subheader(
                "🔴 Detected Flaw Time Ranges"
            )


            for _, flaw in flaws.iterrows():

                severity = flaw["severity"]


                st.write(
                    f"⚠️ Flaw "
                    f"{int(flaw['region_id'])}: "
                    f"{flaw['start_time']:.2f}s → "
                    f"{flaw['end_time']:.2f}s | "
                    f"Duration: "
                    f"{flaw['duration']:.2f}s | "
                    f"Severity: {severity}"
                )


        # =================================================
        # TRANSCRIPT GROUNDING
        # =================================================

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


        # =================================================
        # DETAILED FLAW ANALYSIS
        # =================================================

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


                if severity == "High":

                    icon = "🔴"

                elif severity == "Medium":

                    icon = "🟠"

                else:

                    icon = "🟢"


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
                        f"🎯 **Dominant Acoustic Feature:** "
                        f"{explanation['dominant_feature']}"
                    )


                    st.write(
                        f"📊 **Average Feature Difference:** "
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


                    if (
                        "mathematical_rationale"
                        in explanations.columns
                    ):

                        st.info(
                            f"🧮 **Mathematical Rationale:** "
                            f"{explanation['mathematical_rationale']}"
                        )


                    if (
                        "explanation"
                        in explanations.columns
                    ):

                        st.write(
                            f"🔍 **Explanation:** "
                            f"{explanation['explanation']}"
                        )


                    if (
                        "feedback"
                        in explanations.columns
                    ):

                        st.success(
                            f"💡 **Actionable Feedback:** "
                            f"{explanation['feedback']}"
                        )


        # =================================================
        # DOWNLOAD RESULTS
        # =================================================

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

        st.warning(
            "⚠️ Analysis result files were not generated."
        )


# =========================================================
# NO ANALYSIS YET
# =========================================================

else:

    st.info(
        "👋 Record or upload an Ideal Speech and a Flawed Speech, "
        "then click **Analyze Speech** to see the results."
    )