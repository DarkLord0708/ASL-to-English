# Ambidextrous ASL Fingerspelling Engine

A real-time American Sign Language (ASL) alphabet recognition system built with Python. It uses MediaPipe for lightweight hand-tracking and a Random Forest Classifier to predict static signs. The pipeline includes a "hold-to-type" interface that allows users to spell out full sentences live on screen.

## Key Features
* **Ambidextrous Support:** The training pipeline mathematically mirrors the Kaggle ASL dataset across the X-axis, allowing the model to recognize both left and right-handed signers seamlessly.
* **Temporal Smoothing Buffer:** Utilizes a rolling `deque` buffer and majority-voting to eliminate frame-by-frame prediction flickering.
* **Hold-to-Type Interface:** Features a visual timer that locks in a character when held for 1 second, supporting functional signs like `space` and `del`.
* **Coordinate Normalization:** Extracts 63 3D joint coordinates and normalizes them for translation and scale invariance, making the model robust to distance and screen position.

## Architecture
1. `hand_tracker.py` - Modular wrapper for the MediaPipe Tasks API.
2. `main.py` - Custom webcam data collection script to build personalized CSV datasets.
3. `kaggle_test_creator.py` - Batch processes the Kaggle image dataset into coordinate data.
4. `train_model.py` - Augments coordinate data and trains the Scikit-Learn Random Forest.
5. `inference.py` - Lightweight diagnostic tool for real-time model testing.
6. `make_sentence.py` - The live OpenCV interface featuring the temporal buffer and typing engine.

## Setup & Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

## Dataset & Data Engineering
This model is trained on the [Kaggle ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet), which contains 87,000 images of fingerspelled letters. 

Rather than training a heavy Convolutional Neural Network (CNN) directly on the images, this pipeline uses MediaPipe to extract 63 structural joint coordinates from each image. This approach provides significant advantages:
* **Efficiency:** The extracted coordinates are saved to a lightweight CSV, allowing the Random Forest classifier to train in seconds rather than hours.
* **Ambidextrous Augmentation:** During training, the pipeline duplicates the dataset and mathematically inverts the X-axis coordinates. This synthetically generates a left-handed dataset from the right-handed Kaggle images, making the final model fully ambidextrous.
* **Exclusions:** The pipeline automatically filters out dynamic letters ('J' and 'Z') as they require temporal trajectory tracking, focusing strictly on static shape recognition.

## Building a Custom Dataset
If you want to train the model on your own hand or add custom gestures, you can build a personalized dataset directly from your webcam:

1. Run `main.py` to launch the data collection interface.
2. The script will open your webcam and track your hand using MediaPipe.
3. Press the corresponding letter key on your keyboard (e.g., press `A`) while holding that sign. The script will capture the 63 normalized coordinates and append them to a local `asl_dataset.csv` file.
4. Repeat this from various distances and angles to build a robust dataset.
5. Open `train_model.py` and change the target CSV from `kaggle_asl_dataset.csv` to your newly generated `asl_dataset.csv`, then run the training script to compile your custom model.
6. Test your custom model's raw accuracy using `inference.py` before loading it into the `make_sentence.py` typing interface.
