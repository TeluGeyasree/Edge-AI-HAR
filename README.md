# Edge AI Framework for Real-Time Human Activity Recognition Using Lightweight Deep Learning Models

Android app that recognizes Walking (incl. stairs), Jogging, Sitting and Standing from the phone's
accelerometer and gyroscope. A small 1D-CNN (about 12k parameters) runs fully offline on the phone
using TensorFlow Lite. Final-year project by [add your name and team here].

## What is in this repo
- `notebooks/` preprocessing, model training, cross-validation, TFLite conversion, own-data analysis
- `models/` trained models (Keras and TFLite)
- `android_app/` Kotlin + Jetpack Compose app
- `data/own/` 20 recordings (60 s each, phone in pocket)

## Dataset
WISDM 2019 (UCI repository, dataset 507): https://archive.ics.uci.edu/dataset/507
It is not included here. Download it and run the notebooks in order.
