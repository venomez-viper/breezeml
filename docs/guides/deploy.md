# Deploy: DataFrame to API in one line

`deploy()` writes a complete, self-contained serving directory for any
trained BreezeML model:

```python
import breezeml
from breezeml import datasets

df = datasets.iris()
model = breezeml.fit(df, "species")

breezeml.deploy(model, "iris_api", name="iris-classifier")
```

```
iris_api/
|-- model.joblib      # raw sklearn pipeline - no breezeml at runtime
|-- app.py            # FastAPI app: /predict, /health, /docs
|-- requirements.txt  # fastapi, uvicorn, scikit-learn, pandas, joblib
|-- Dockerfile        # python:3.11-slim container
`-- README.md         # copy-paste run instructions
```

## Run it

```bash
cd iris_api
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

Or containerized:

```bash
docker build -t iris-classifier .
docker run -p 8000:8000 iris-classifier
```

## Call it

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '[{"sepal length (cm)": 5.1, "sepal width (cm)": 3.5,
        "petal length (cm)": 1.4, "petal width (cm)": 0.2}]'
# {"predictions": [0]}
```

The app validates feature columns and returns a 422 with the missing column
names when the payload is wrong. Interactive Swagger docs live at `/docs`.

## Zero lock-in, again

`deploy()` saves the **raw sklearn pipeline**, not a BreezeML object. The
serving app never imports breezeml. If you delete BreezeML tomorrow, your
API keeps running.

## ONNX (optional)

```python
from breezeml.deploy import to_onnx
to_onnx(model, "model.onnx")   # pip install breezeml[onnx]
```

Currently supports numeric-only pipelines (no one-hot encoded categorical
columns).

## When NOT to use it

The generated app is a solid starting point, not a hardened production
gateway: add auth, rate limiting, and monitoring before exposing it to the
internet.
