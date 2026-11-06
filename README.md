# Edge AI Framework for Real-Time Human Activity Recognition Using Lightweight Deep Learning Models

A personal project: an Android app that recognizes **Walking (incl. stairs), Jogging, Sitting and Standing**
from the phone's accelerometer and gyroscope. A small 1D-CNN (about 12k parameters) runs fully offline
on the phone with TensorFlow Lite. The app also counts steps and keeps a daily log, a weekly chart,
goals and a sitting reminder.

## What I did
- Built the data pipeline: merged phone accelerometer and gyroscope files by timestamp, fixed uneven sampling
  (resampled to a uniform 20 Hz) and cut 5-second windows (100 x 6)
- Evaluated by person: no subject appears in both training and test data, plus 5-fold cross-validation
- Compared three small models (1D-CNN, GRU, CNN-LSTM), with and without the gyroscope
- Recorded 20 minutes of my own data with the app and used it to fix a real problem (see below)
- Compressed the model with TensorFlow Lite and deployed it in an Android app (Kotlin, Jetpack Compose)

## Results
| Setting | Result |
|---|---|
| 5 classes, 5-fold person-wise cross-validation (CNN) | 78 - 80% accuracy and macro F1 |
| 4 classes, 9 unseen WISDM people | 89 - 92% |
| 4 classes, my held-out recordings (one person, so optimistic) | 99 - 100% |
| TFLite size | 50 KB float32, 28 KB float16, 21 KB INT8 (accuracy changed by 0.3 points or less) |
| Step counting, last two 100-step walks | my detector 96 and 96, phone's built-in sensor 99 and 94 |

## What I learned
- The model first read my jogging as Stairs on my own phone. My jogging was milder than the dataset's,
  so I recorded my own data and retrained, which fixed it. Models trained on one dataset can fail on a new phone and person.
- Single train/test splits move by several points between runs, so the cross-validation numbers are the ones to trust.
- The phone's built-in step sensor was inconsistent on my phone (37% to 118% of the true count on different walks).

## Known limits
- Walking and Stairs cannot be separated, so Stairs counts as Walking.
- Shaking the phone in the hand is read as Jogging and adds false steps to every step counter I tested.
- The app only counts while it is open with the screen on.
- Everything was measured with one phone (OnePlus CPH2381); the live checks come from screen recordings, not a formal test set.

## Repository
- `notebooks/` 01 preprocessing, 02 model training, 03 TFLite optimization, 04 cross-validation, 05 own-data analysis
- `models/` trained models (Keras and TFLite) and normalization values
- `android_app/` the app
- `data/own/` my 20 recordings (60 s each, phone in pocket)
- `figures/` confusion matrices and result tables

## Dataset
WISDM Smartphone and Smartwatch Activity and Biometrics Dataset (UCI, dataset 507):
https://archive.ics.uci.edu/dataset/507. It is not included here; download it and run the notebooks in order.

## Run
Notebooks: Python 3.9 and TensorFlow 2.19. App: open `android_app/` in Android Studio.
