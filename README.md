# AI Content Detection Prototype

This project contains a lightweight content detector prototype that can run as a pure-Python fallback heuristic and optionally use a Hugging Face transformer path when available.

## Files
- `detector.py`: trained-model scoring plus transformer and heuristic fallbacks
- `app.py`: minimal Flask web UI and JSON API wrapper
- `train_model.py`: trains a text classifier using scikit-learn and saves the model plus metrics
- `calibrate.py`: derives the score calibration from out-of-fold cross-validation
- `calibration.json`: the calibration artifact the detector loads at runtime
- `dataset.json`: small labeled sample dataset for training and evaluation
- `requirements_ml.txt`: dependencies for the trained model pipeline
- `tests/test_detector.py`: unit tests for the scoring contract
- `pytest.ini`: pytest configuration to load local modules automatically

## How scoring works

The score is a **calibrated estimate of P(not written by a human)**, 0-100. The
label is derived from that score and from nothing else, in a single function
(`detector._finalise`), so the two can never disagree.

| Label | Meaning |
| --- | --- |
| `Low risk` | Below the review band |
| `Needs review` | Above the threshold that mislabels 20% of known human samples |
| `High risk` | Above the threshold that mislabels 5% of known human samples |
| `Inconclusive` | The detector declined to judge. `score` is `null` |

The bands are not hand-picked. `calibrate.py` sets them at measured
false-positive rates on human writing, because the cost of wrongly flagging a
student is much higher than the cost of missing a generated one.

### When the detector abstains

`score` comes back as `null` and the label is `Inconclusive` when:

- the submission is under 50 words (override with `DETECTION_MIN_WORDS`)
- the text is empty
- `calibration.json` is missing or unreadable

Abstaining is a deliberate answer, not an error. Raw `predict_proba` output on a
two-word input is a three-way coin flip, and reporting that as a risk score is
how the earlier version returned "High risk, 65.2" for the input `ok`.

### Calibrated vs advisory engines

Only the trained-model path is calibrated. The transformer and keyword-heuristic
fallbacks are marked `advisory: true` in the response and **cannot return
`High risk`** - the most they can say is `Needs review`. An uncalibrated keyword
match is not a sound basis for an accusation.

If the running scikit-learn version does not match the one recorded in
`calibration.json`, results are downgraded to advisory as well. A model pickled
under one version and scored under another shifted the raw signal by up to 0.04,
which the fitted slope turns into a double-digit score change.

### Regenerating the calibration

After changing `dataset.json`, the pipeline, or the pinned scikit-learn version:

```bash
python train_model.py && python calibrate.py && pytest -q
```

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
- `metrics.json`: precision, recall, and F1-score for a single hold-out split

## Reported performance

Use the **out-of-fold figures in `calibration.json`**, not the ones in
`metrics.json`:

| Measure | Value |
| --- | --- |
| Out-of-fold macro F1 (6-fold) | **0.713** |
| Spread across folds | **0.252** |
| AUC, human vs generated | 0.960 |
| Median score, human samples | 11.9 |
| Median score, generated samples | 97.7 |

`metrics.json` reports a higher number (0.944 F1). That figure comes from a
single 18-sample hold-out split of a 252-sample multi-domain dataset and moves by several
points if you change `random_state`. It is kept for reference and carries a
`health_warning` field, but it is not a generalisable accuracy claim.

The fold spread of 0.169 is the honest headline: on 72 samples with a median
length of 21 words, any single accuracy number is noise.

## Testing
Run the unit tests from the repo root:

```bash
pytest -q
```

## Notes
- The detector is designed to be lightweight and usable as a service for sites or Moodle integration.
- The heuristics path works without heavy ML dependencies.
- The transformer path is optional and only enabled when `transformers` and `torch` are installed.
- Dependencies in `requirements.txt` are pinned on purpose. The calibration is
  fitted against a specific scikit-learn version; bump them together and re-run
  `train_model.py` and `calibrate.py`.

## Limitations

Read this before showing anyone a score.

- **252 diverse academic training samples.** The corpus spans literature, history, philosophy, STEM, economics, psychology, and international student writing.
  indicative. It is calibrated on short snippets but used on whole essays.
- **A score is not proof.** It is one piece of evidence for a human reviewer to
  weigh alongside drafts, version history, and a conversation with the student.
- **Known bias risk.** AI detectors are documented to over-flag writing by
  non-native English speakers. The held-out non-native sample in the test suite
  currently scores low, but three human samples are not evidence of fairness.
- **Production Hardening.** Includes 10MB upload limit (`MAX_CONTENT_LENGTH`), `defusedxml` DOCX parsing security, and Gunicorn WSGI production deployment.
- This is a prototype intended for demonstration and project submission, not for
  production deployment against real student submissions.
