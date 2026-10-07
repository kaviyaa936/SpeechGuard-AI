import soundfile as sf
import numpy as np
import pandas as pd
import os


def extract_features(audio_path, output_path):

    # =================================================
    # LOAD AUDIO
    # =================================================

    y, sr = sf.read(audio_path)

    # Convert stereo to mono
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)

    y = y.astype(np.float64)


    # =================================================
    # PARAMETERS
    # =================================================

    frame_length = 2048
    hop_length = 512

    if len(y) < frame_length:
        y = np.pad(
            y,
            (0, frame_length - len(y))
        )


    # =================================================
    # CREATE FRAMES
    # =================================================

    num_frames = 1 + (
        len(y) - frame_length
    ) // hop_length

    frames = np.zeros(
        (frame_length, num_frames)
    )

    for i in range(num_frames):

        start = i * hop_length
        end = start + frame_length

        frames[:, i] = y[start:end]


    # =================================================
    # HANNING WINDOW
    # =================================================

    window = np.hanning(
        frame_length
    )

    frames = frames * window[:, None]


    # =================================================
    # 1. ENERGY
    # =================================================

    energy = np.sqrt(
        np.mean(
            frames ** 2,
            axis=0
        )
    )


    # =================================================
    # 2. ZERO CROSSING RATE
    # =================================================

    signs = np.sign(frames)

    signs[signs == 0] = 1

    zcr = np.mean(
        np.abs(
            np.diff(
                signs,
                axis=0
            )
        ),
        axis=0
    ) / 2


    # =================================================
    # 3. FFT
    # =================================================

    fft_complex = np.fft.rfft(
        frames,
        axis=0
    )

    fft_magnitude = np.abs(
        fft_complex
    )


    # =================================================
    # FREQUENCY AXIS
    # =================================================

    frequencies = np.fft.rfftfreq(
        frame_length,
        d=1 / sr
    )


    # =================================================
    # 4. SPECTRAL CENTROID
    # =================================================

    spectral_centroid = (
        np.sum(
            frequencies[:, None]
            * fft_magnitude,
            axis=0
        )
        /
        (
            np.sum(
                fft_magnitude,
                axis=0
            )
            + 1e-10
        )
    )


    # =================================================
    # 5. DOMINANT FREQUENCY
    # =================================================

    dominant_indices = np.argmax(
        fft_magnitude,
        axis=0
    )

    dominant_frequency = frequencies[
        dominant_indices
    ]


    # =================================================
    # 6. PITCH / F0
    # =================================================

    pitch = []

    min_lag = max(
        1,
        int(sr / 500)
    )

    max_lag = min(
        frame_length - 1,
        int(sr / 70)
    )


    for frame in frames.T:

        correlation = np.correlate(
            frame,
            frame,
            mode="full"
        )

        correlation = correlation[
            len(correlation) // 2:
        ]

        if max_lag <= min_lag:

            pitch.append(0)

            continue


        search_region = correlation[
            min_lag:max_lag
        ]

        peak = (
            np.argmax(
                search_region
            )
            + min_lag
        )


        if (
            correlation[peak]
            >
            0.3 * correlation[0]
        ):

            pitch.append(
                sr / peak
            )

        else:

            pitch.append(0)


    pitch = np.array(
        pitch
    )


    # =================================================
    # 7. MFCC
    # =================================================

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

    high_mel = hz_to_mel(
        sr / 2
    )


    mel_points = np.linspace(
        low_mel,
        high_mel,
        n_mels + 2
    )


    hz_points = mel_to_hz(
        mel_points
    )


    bin_points = np.floor(
        (frame_length + 1)
        * hz_points
        / sr
    ).astype(int)


    power_spectrum = (
        fft_magnitude ** 2
    ) / frame_length


    # =================================================
    # MEL FILTER BANK
    # =================================================

    mel_filters = np.zeros(
        (
            n_mels,
            len(frequencies)
        )
    )


    for m in range(
        1,
        n_mels + 1
    ):

        left = bin_points[
            m - 1
        ]

        center = bin_points[
            m
        ]

        right = bin_points[
            m + 1
        ]


        if center > left:

            mel_filters[
                m - 1,
                left:center
            ] = (
                np.arange(
                    left,
                    center
                ) - left
            ) / (
                center - left
            )


        if right > center:

            mel_filters[
                m - 1,
                center:right
            ] = (
                right
                - np.arange(
                    center,
                    right
                )
            ) / (
                right - center
            )


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


    # =================================================
    # DCT-II
    # =================================================

    n = np.arange(
        n_mels
    )


    mfcc = np.zeros(
        (
            n_mfcc,
            log_mel_energy.shape[1]
        )
    )


    for k in range(
        n_mfcc
    ):

        cosine = np.cos(
            np.pi
            * k
            * (2 * n + 1)
            / (2 * n_mels)
        )

        mfcc[k] = np.sum(
            log_mel_energy
            * cosine[:, None],
            axis=0
        )


    # =================================================
    # TIME
    # =================================================

    times = (
        np.arange(
            num_frames
        )
        * hop_length
        / sr
    )


    # =================================================
    # FEATURE TABLE
    # =================================================

    features = pd.DataFrame({

        "time": times,

        "energy": energy,

        "zcr": zcr,

        "spectral_centroid":
            spectral_centroid,

        "dominant_frequency":
            dominant_frequency,

        "pitch_f0":
            pitch
    })


    # =================================================
    # ADD MFCC FEATURES
    # =================================================

    for i in range(
        n_mfcc
    ):

        features[
            f"mfcc_{i + 1}"
        ] = mfcc[i]


    # =================================================
    # SAVE
    # =================================================

    output_folder = os.path.dirname(
        output_path
    )


    if output_folder:

        os.makedirs(
            output_folder,
            exist_ok=True
        )


    features.to_csv(
        output_path,
        index=False
    )


    # =================================================
    # INFORMATION
    # =================================================

    print(
        "Feature extraction successful!"
    )

    print(
        "Audio:",
        audio_path
    )

    print(
        "Sample rate:",
        sr
    )

    print(
        "Duration:",
        round(
            len(y) / sr,
            2
        ),
        "seconds"
    )

    print(
        "Number of frames:",
        len(features)
    )

    print(
        "FFT dominant frequency feature added!"
    )

    print(
        "Saved to:",
        output_path
    )


    return len(y) / sr


# =================================================
# TEST
# =================================================

if __name__ == "__main__":

    extract_features(
        "data/speech_001/ideal/ideal_001.wav",
        "results/ideal_001_features.csv"
    )

    extract_features(
        "data/speech_001/flawed/flawed_001.wav",
        "results/flawed_001_features.csv"
    )

    print(
        "\nBoth ideal and flawed features are ready!"
    )