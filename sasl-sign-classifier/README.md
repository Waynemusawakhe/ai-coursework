# South African Sign Language Letter Classifier

An image classifier trained with Google Teachable Machine to recognise hand signs for letters of the South African Sign Language alphabet (CMPG313, 2025).

## What I did

- Collected the training images myself: both hands, several angles, a plain background and even lighting
- Trained one class per letter and tested it live on the webcam

## What I found

- More images and more variety per class (hands, angles) clearly improved accuracy.
- Letters with very similar hand shapes, such as **A** and **E** (both a closed fist), were confused: the model split its probability between the two.
- Shadows and motion blur in the training images lowered recognition accuracy.
- Teachable Machine is fast for prototyping but offers little control for fine-tuning.

## Planned rebuild

Rebuild in Python with MediaPipe hand landmarks (21 key points per hand) feeding a small classifier. Working from finger geometry instead of raw pixels should separate look-alike letters such as A and E, and stop lighting and background from mattering. The aim is a live webcam demo for assistive and educational tools.
