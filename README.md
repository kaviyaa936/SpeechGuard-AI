# SpeechGuard AI

## Multimodal AI Hackathon 2026

SpeechGuard AI is an AI-powered speech analysis system that compares ideal and flawed speech recordings to detect temporal speech flaws and explain their possible acoustic causes.

## Problem

Traditional speech analysis often provides only an overall score. It does not clearly show:

* When a speech flaw occurred
* How long the flaw lasted
* What acoustic feature changed
* How severe the flaw was
* Which part of the transcript was affected

SpeechGuard AI addresses these limitations through temporal and feature-level speech analysis.

## Key Features

* Ideal vs flawed speech comparison
* Audio feature extraction
* Temporal flaw detection
* Exact time-range grounding
* Feature-specific causal explanations
* Severity classification
* Speech quality score
* Transcript grounding
* Interactive Streamlit dashboard
* CSV result downloads

## Audio Features

The system extracts:

* Energy
* Zero Crossing Rate (ZCR)
* Spectral Centroid
* Pitch / F0
* 13 MFCC features

## System Pipeline

```text
Ideal Speech
     |
     v
Feature Extraction
     |
     +----------------------+
                            |
                            v
                    Feature Comparison
                            |
Flawed Speech               v
     |              Temporal Flaw Detection
     |                       |
     +-----------------------+
                             |
                             v
                  Causal Explanation
                             |
                             v
                     Rubric Scoring
                             |
                             v
                   Streamlit Dashboard
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
│
├── models/
│
└── README.md
```

## Technology Stack

* Python
* Pandas
* NumPy
* Librosa
* Streamlit
* PyAV
* CSV-based analysis pipeline

## Output
## Demo

SpeechGuard AI analyzes ideal and flawed speech and provides:

* Temporal flaw detection with exact time ranges
* Severity classification
* Feature-based causal explanations
* Rubric-based speech score
* Transcript-based flaw grounding
* Interactive Streamlit dashboard
* Downloadable analysis results

### Example Analysis

The system compares ideal and flawed speech, detects deviations over time, identifies important flaw regions, and explains the dominant acoustic feature responsible for each detected flaw.

Example output:

* Overall Score: 40/100
* Rating: Poor
* Detected Flaws: 9
* High Severity: 5
* Medium Severity: 4
* Low Severity: 0


The system generates:

* Temporal flaw regions
* Start and end timestamps
* Flaw duration
* Dominant acoustic feature
* Feature deviation
* Severity level
* Causal explanation
* Speech quality score

## Running the Application

Activate the virtual environment and run:

```bash
streamlit run app.py
```

Then open the Streamlit URL shown in the terminal.

## Important Note

For plain text transcripts without timestamps, transcript grounding is performed at the paragraph/segment level using the detected flaw position.

Exact word-level grounding requires a timestamped transcript or forced alignment system.

## Future Improvements

* Automatic speech recognition
* Word-level timestamp alignment
* Automatic pronunciation error classification
* Real-time speech analysis
* Advanced AI-based causal reasoning
* Improved acoustic anomaly detection
* Speaker-independent evaluation

## Project Goal

SpeechGuard AI aims to make speech evaluation more explainable by showing not only **whether** a speech problem exists, but also **when it occurs, what changed, and why it may have happened**.
