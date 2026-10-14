# SpeechGuard AI

## Multimodal AI Hackathon 2026

SpeechGuard AI is an AI-powered speech analysis system that compares ideal and flawed speech recordings to detect temporal speech flaws and explain their possible acoustic causes.

## Problem Statement

Traditional speech analysis systems often provide only an overall score without clearly explaining where and why speech flaws occur.

They may not clearly identify:

- When a speech flaw occurred
- How long the flaw lasted
- Which acoustic feature changed
- How severe the flaw was
- Which part of the transcript was affected

SpeechGuard AI addresses these limitations through temporal and feature-level speech analysis.

## Key Features

- Ideal vs. flawed speech comparison
- Audio feature extraction
- Temporal flaw detection
- Time-range grounding of detected flaws
- Feature-based causal explanations
- Severity classification
- Rubric-based speech quality scoring
- Transcript-based flaw grounding
- Interactive Streamlit dashboard
- Downloadable CSV analysis results

## Audio Features

The system extracts and compares the following acoustic features:

- Energy
- Zero Crossing Rate (ZCR)
- Spectral Centroid
- Pitch / Fundamental Frequency (F0)
- 13 Mel-Frequency Cepstral Coefficients (MFCCs)

These features help identify acoustic differences between ideal and flawed speech.

## System Pipeline

```text
Ideal Speech                 Flawed Speech
     |                            |
     v                            v
Feature Extraction          Feature Extraction
     |                            |
     +-------------+--------------+
                   |
                   v
           Feature Comparison
                   |
                   v
         Temporal Flaw Detection
                   |
                   v
          Causal Explanation
                   |
                   v
             Rubric Scoring
                   |
                   v
          Streamlit Dashboard
                   |
                   v
           CSV Result Downloads
```

## Project Structure

```text
SpeechGuard-AI/
│
├── app.py
│
├── data/
│   └── speech_001/
│       ├── ideal/
│       ├── flawed/
│       ├── transcript/
│       └── alignment/
│
├── audio/
│
├── src/
│   ├── __init__.py
│   ├── feature_extraction.py
│   ├── compare_features.py
│   ├── flaw_detection.py
│   ├── causal_explanation.py
│   └── rubric_score.py
│
├── results/
│   ├── ideal_001_features.csv
│   ├── flawed_001_features.csv
│   ├── uploaded_ideal_features.csv
│   ├── uploaded_flawed_features.csv
│   ├── feature_comparison.csv
│   ├── detected_flaws.csv
│   ├── causal_explanations.csv
│   └── rubric_score.csv
│
├── docs/
├── models/
├── requirements.txt
└── README.md
```

## Technology Stack

- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy
- **Audio Processing:** SoundFile, PyAV
- **Acoustic Analysis:** Audio feature extraction and comparison
- **Web Application:** Streamlit
- **Results:** CSV-based analysis pipeline

## Output and Demo

SpeechGuard AI analyzes ideal and flawed speech recordings and presents the results through an interactive Streamlit dashboard.

The system provides:

- Detected flaw regions with start and end timestamps
- Flaw duration
- Acoustic feature deviations
- Severity levels
- Feature-based causal explanations
- Rubric-based speech quality score
- Transcript-based flaw grounding
- Downloadable analysis results

### Example Analysis

For the current example analysis, the system produced the following results:

- **Overall Score:** 78/100
- **Rating:** Good
- **Detected Flaws:** 5
- **High Severity:** 0
- **Medium Severity:** 4
- **Low Severity:** 1

These values represent the current example run and may change when different recordings or analysis settings are used.

The generated results include temporal flaw regions, timestamps, duration, dominant acoustic features, feature deviations, severity levels, and causal explanations.

## Installation and Setup

### Prerequisites

- Python installed on the system
- Git
- A compatible web browser

### 1. Clone the Repository

```bash
git clone https://github.com/kaviyaa936/SpeechGuard-AI.git
cd SpeechGuard-AI
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

**Windows:**

```bash
.venv\Scripts\activate
```

### 4. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Run the Application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal to access the Streamlit dashboard.

## Important Note on Transcript Grounding

For plain-text transcripts without timestamps, transcript grounding is performed at the paragraph or segment level using the detected flaw position.

Exact word-level grounding requires a timestamped transcript or a forced-alignment system. Word-level alignment should not be assumed when timestamp information is unavailable.

## Future Improvements

- Automatic speech recognition
- Word-level timestamp alignment
- Automatic pronunciation error classification
- Real-time speech analysis
- Advanced AI-based causal reasoning
- Improved acoustic anomaly detection
- Speaker-independent evaluation

## Project Goal

SpeechGuard AI aims to make speech evaluation more explainable by showing not only **whether** a speech problem exists, but also **when it occurs, what changed, and why it may have happened**.
