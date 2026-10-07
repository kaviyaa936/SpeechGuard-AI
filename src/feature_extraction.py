import librosa
import numpy as np
import pandas as pd
import os


def extract_features(audio_path, output_path):

    # Load audio
    y, sr = librosa.load(audio_path, sr=None)

    frame_length = 2048
    hop_length = 512

    # Create frames
    frames = librosa.util.frame(
        y,
        frame_length=frame_length,
        hop_length=hop_length
    )

    # Window
    window = np.hanning(frame_length)
    frames = frames * window[:, None]

    # -----------------------------
    # 1. Energy
    # -----------------------------
    energy = np.sqrt(np.mean(frames ** 2, axis=0))

    # -----------------------------
    # 2. Zero Crossing Rate
    # -----------------------------
    zcr = np.mean(
        np.abs(np.diff(np.sign(frames), axis=0)),
        axis=0
    ) / 2

    # -----------------------------
    # 3. Spectral Centroid
    # -----------------------------
    fft_values = np.abs(np.fft.rfft(frames, axis=0))

    frequencies = np.fft.rfftfreq(
        frame_length,
        1 / sr
    )

    spectral_centroid = (
        np.sum(
            frequencies[:, None] * fft_values,
            axis=0
        )
        / (np.sum(fft_values, axis=0) + 1e-10)
    )

    # -----------------------------
    # 4. Pitch / F0
    # -----------------------------
    pitch = []

    for frame in frames.T:

        correlation = np.correlate(
            frame,
            frame,
            mode="full"
        )

        correlation = correlation[
            len(correlation) // 2:
        ]

        min_lag = max(1, int(sr / 500))

        max_lag = min(
            len(correlation) - 1,
            int(sr / 70)
        )

        if max_lag <= min_lag:
            pitch.append(0)
            continue

        peak = (
            np.argmax(
                correlation[min_lag:max_lag]
            )
            + min_lag
        )

        if correlation[peak] > 0.3 * correlation[0]:
            pitch.append(sr / peak)
        else:
            pitch.append(0)

    pitch = np.array(pitch)

    # -----------------------------
    # 5. MFCC
    # -----------------------------

    def hz_to_mel(hz):
        return 2595 * np.log10(
            1 + hz / 700
        )

    def mel_to_hz(mel):
        return 700 * (
            10 ** (mel / 2595) - 1
        )

    n_mels = 26
    n_mfcc = 13

    low_mel = hz_to_mel(0)
    high_mel = hz_to_mel(sr / 2)

    mel_points = np.linspace(
        low_mel,
        high_mel,
        n_mels + 2
    )

    hz_points = mel_to_hz(mel_points)

    bin_points = np.floor(
        (frame_length + 1)
        * hz_points
        / sr
    ).astype(int)

    power_spectrum = (
        fft_values ** 2
    ) / frame_length

    mel_filters = np.zeros(
        (n_mels, len(frequencies))
    )

    for m in range(1, n_mels + 1):

        left = bin_points[m - 1]
        center = bin_points[m]
        right = bin_points[m + 1]

        if center > left:
            mel_filters[
                m - 1,
                left:center
            ] = (
                np.arange(left, center) - left
            ) / (center - left)

        if right > center:
            mel_filters[
                m - 1,
                center:right
            ] = (
                right - np.arange(center, right)
            ) / (right - center)

    mel_energy = np.dot(
        mel_filters,
        power_spectrum
    )

    mel_energy = np.maximum(
        mel_energy,
        1e-10
    )

    log_mel_energy = np.log(
        mel_energy
    )

    # DCT-II
    n = np.arange(n_mels)

    mfcc = np.zeros(
        (
            n_mfcc,
            log_mel_energy.shape[1]
        )
    )

    for k in range(n_mfcc):

        mfcc[k] = np.sum(
            log_mel_energy
            * np.cos(
                np.pi
                * k
                * (2 * n + 1)
                / (2 * n_mels)
            )[:, None],
            axis=0
        )

    # -----------------------------
    # Create feature table
    # -----------------------------

    times = (
        np.arange(len(energy))
        * hop_length
        / sr
    )

    features = pd.DataFrame({
        "time": times,
        "energy": energy,
        "zcr": zcr,
        "spectral_centroid": spectral_centroid,
        "pitch_f0": pitch
    })

    for i in range(n_mfcc):
        features[
            f"mfcc_{i + 1}"
        ] = mfcc[i]

    # Create output folder
    output_folder = os.path.dirname(output_path)

    if output_folder:
        os.makedirs(
            output_folder,
            exist_ok=True
        )

    features.to_csv(
        output_path,
        index=False
    )

    print("Feature extraction successful!")
    print("Audio:", audio_path)
    print("Sample rate:", sr)
    print(
        "Duration:",
        round(len(y) / sr, 2),
        "seconds"
    )
    print(
        "Number of frames:",
        len(features)
    )
    print("Saved to:", output_path)

    return len(y) / sr


# ==================================================
# OPTIONAL: TEST WITH EXISTING SAMPLE AUDIO
# ==================================================

if __name__ == "__main__":

    extract_features(
        "data/speech_001/ideal/ideal_001.wav",
        "results/ideal_001_features.csv"
    )

    extract_features(
        "data/speech_001/flawed/flawed_001.wav",
        "results/flawed_001_features.csv"
    )

    print("\nBoth ideal and flawed features are ready!")