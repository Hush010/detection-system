# AI Content Detection Prototype

This project contains a lightweight content detector prototype that can run as a pure-Python fallback heuristic and optionally use a Hugging Face transformer path when available.

## Files
- `detector.py`: fallback heuristic detector and optional Hugging Face transformer path
- `app.py`: minimal Flask web UI and JSON API wrapper
- `train_model.py`: trains a text classifier using scikit-learn and saves the model plus metrics
- `dataset.json`: small labeled sample dataset for training and evaluation
- `requirements_ml.txt`: dependencies for the trained model pipeline
- `tests/test_detector.py`: unit tests for the fallback logic
- `pytest.ini`: pytest configuration to load local modules automatically

## Run the detector
### Lightweight usage
Install only Flask if you want the web/API wrapper:

```bash
pip install Flask
```

Start the app:

```bash
python app.py
```

Then visit:
- `http://localhost:5000/` for the web UI
- `POST http://localhost:5000/api/analyze` with JSON `{"text": "..."}` for API use
- `GET http://localhost:5000/api/health` for a simple health check

### Full model and training
To use the training pipeline or the transformer-enhanced detection path, install the ML dependencies:

```bash
pip install -r requirements_ml.txt
```

Then train the classifier:

```bash
python train_model.py
```

This produces:
- `model.joblib`: trained text classifier
- `metrics.json`: precision, recall, and F1-score

## Testing
Run the unit tests from the repo root:

```bash
pytest -q
```

## Notes
- The detector is designed to be lightweight and usable as a service for sites or Moodle integration.
- The heuristics path works without heavy ML dependencies.
- The transformer path is optional and only enabled when `transformers` and `torch` are installed.
- This is a prototype intended for demonstration and project submission, not for production deployment.
