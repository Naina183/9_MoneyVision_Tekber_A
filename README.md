# MoneyVision

MoneyVision is a computer vision prototype built with OpenCV to help blind and low-vision users identify Indonesian rupiah banknotes, calculate cash totals, and check change through Indonesian voice feedback.

## Features

- **Banknote Recognition:** Identifies supported rupiah banknote denominations using image processing and feature matching.
- **Multiple Banknotes:** Detects multiple banknotes in a single frame.
- **Cash Calculation:** Calculates the total amount and expected change based on the purchase price.
- **Change Verification:** Compares received change with the expected amount.
- **Voice Feedback:** Announces recognized denominations and transaction results in Indonesian.
- **Flexible Input:** Supports webcam, images, and video files.

## Supported Banknotes

Rp1,000, Rp2,000, Rp5,000, Rp10,000, Rp20,000, Rp50,000, and Rp100,000.

The prototype supports selected banknote designs from the 2016 and 2022 rupiah series, including front and back sides.

## Tech Stack

- Python
- OpenCV
- NumPy
- ORB Feature Matching
- RANSAC Homography Verification
- pyttsx3 / Google Text-to-Speech

## Getting Started

### Requirements

- Python 3.9 or later
- A webcam for real-time use (optional)

### Installation

Clone the repository and install the dependencies:

```bash
git clone <repository-url>
cd MoneyVision
python -m venv .venv
```

Activate the virtual environment:

**Windows**

```bash
.venv\Scripts\activate
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

### Run the Application

Use a webcam:

```bash
python main.py
```

Recognize banknotes from an image:

```bash
python main.py --image samples/scene_demo_meja.jpg
```

Use Google Text-to-Speech:

```bash
python main.py --tts gtts
```

Google Text-to-Speech requires an internet connection and its optional dependencies.

## How It Works

1. **Detection:** Finds banknotes in an image or video frame.
2. **Preprocessing:** Corrects perspective and prepares banknotes for recognition.
3. **Recognition:** Matches visual features against reference images to identify denominations.
4. **Calculation:** Adds recognized amounts and calculates expected change.
5. **Audio Feedback:** Announces results in Indonesian.

## Limitations

MoneyVision is an experimental prototype and is not intended to replace independent verification during financial transactions.

- Performance depends on lighting, background contrast, camera angle, and image quality.
- Overlapping, folded, or partially visible banknotes may not be detected or recognized correctly.
- Unsupported currencies and banknotes outside the reference set may be rejected.
- The prototype does not verify banknote authenticity.

## Project Status

Developed as an academic Computer Vision project using OpenCV.

## Acknowledgements

This project uses publicly available rupiah banknote image datasets for development and evaluation.

- [OpenCV](https://opencv.org/)
- [Dataset A: Rupiah-PCR-Using-Image-Processing-and-CNN](https://github.com/ardinadnn)
- [Dataset B: rupiah-currency-image-recognition-system](https://github.com/weniariska)
- [Dataset D: rupiah-banknotes-dataset](https://github.com/mkaspulanwar)

See the respective repositories for dataset details and licensing information.