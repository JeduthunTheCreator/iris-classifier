# Iris Classifier (Decision Tree)

## Overview
End‑to‑end ML example from Digital Marketing Mastery Module → builds a decision‑tree classifier on the classic Iris dataset using scikit‑learn.

## Quick start
```bash
git clone https://github.com/<YOUR_USERNAME>/iris-classifier.git
cd iris-classifier
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python src/train.py
```

## Sample command to run Python Script in CLI
```bash
python src/train.py --test-size 0.2 --random-state 42

for more information on the feature importance run
python src/train.py --test-size 0.2 --random-state 42 --verbose
```