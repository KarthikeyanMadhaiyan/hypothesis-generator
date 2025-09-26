# Hypothesis Generator

This project generates likely hypotheses for anomalies using metadata.

## Steps to Run (Without Docker)

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Run generator:
   ```bash
   python hypothesis_generator.py
   ```

3. Check output in `output.json`.

---

## Steps to Run (With Docker)

1. Build image:
   ```bash
   docker build -t hypothesis-generator .
   ```

2. Run container:
   ```bash
   docker run -v %cd%:/app hypothesis-generator
   ```
