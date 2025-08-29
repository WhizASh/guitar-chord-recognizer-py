# Guitar Chord Recognizer

A real-time guitar chord recognition system built with Python, OpenCV, and MediaPipe Hand Detection.

## Features

- Real-time hand position tracking on guitar neck
- Support for common guitar chords (Cmaj, Emaj, Emin, Amaj, Amin, Fmaj)
- Guitar neck calibration system
- Mirror mode for natural playing view
- Visual confidence indicator
- Finger position smoothing for stability
- Debug visualization options

## Prerequisites

- Python 3.7+
- OpenCV (`cv2`)
- MediaPipe
- NumPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/guitar-chord-recognizer.git
cd guitar-chord-recognizer
```

2. Install required packages:
```bash
pip install opencv-python mediapipe numpy
```

3. Download the MediaPipe hand landmark model:
```bash
wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

## Usage

1. Run the main program:
```bash
python main.py
```

2. Choose calibration option:
   - Use existing calibration (option 1)
   - Perform new calibration (option 2)

3. Controls:
   - Press 'ESC' to quit
   - Press 'c' to recalibrate
   - Press 'r' to reset history
   - Press 'm' to toggle mirror mode

## Calibration

When calibrating:
1. Position your guitar in view of the camera
2. Mark the string positions (top to bottom)
3. Mark the fret positions (left to right)
4. Calibration will be saved for future use

## Command Line Options

- `--no-mirror`: Disable mirroring (use raw camera view)
```bash
python main.py --no-mirror
```

## Supported Chords

- Cmaj
- Emaj
- Emin
- Amaj
- Amin
- Fmaj

## How It Works

1. Uses MediaPipe for hand landmark detection
2. Maps finger positions to guitar neck coordinates
3. Matches finger patterns to known chord positions
4. Applies temporal smoothing for stability
5. Provides visual feedback and confidence scoring

## Contributing

Feel free to open issues or submit pull requests for improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for