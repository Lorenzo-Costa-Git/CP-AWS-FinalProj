# Final Exam – Theoretical Questions

## 3.1 API Design

### Question 1
*The request schema for `POST /diagnose` uses cumulative timestamps rather than pre-computed partial times. List the 7 fields, explain why cumulative input is the right choice, and justify why this set is the minimum necessary to produce the response defined in §1.4.*

The `PieceRequest`object (defined in `api/src/app.py`) takes 7 fields:
- `piece_id` — tags request so that it can be matched with response afterwards
- `die_matrix` — taken from reference_times.json, this tells the API which set of reference times to compare against
- `lifetime_2nd_strike_s` — cumulative time at 2nd strike
- `lifetime_3rd_strike_s` — cumulative time at 3rd strike
- `lifetime_4th_strike_s` — cumulative time at 4th strike
- `lifetime_auxiliary_press_s` — cumulative time at auxiliary press
- `lifetime_bath_s` — cumulative time at bath entry

Cumulative timestamps are the right input choice because that is the way PLC records them, throughout the whole duration of the process and not only in individual segments. This is done so that the API can immediately take the data without the need to reformat. This signifies less processing for the client and therefore less opportunities for errors. 

7 fields are the minimum needed as `piece_id` and `die_matrix` are needed to identify the piece and select the correct reference; and the 5 cumulative timestamps are the minimum needed to compute all 5 partial times. 

### Question 2
*Your API loads `reference_times.json` once at startup rather than reading it on every request. Explain why this is the right approach for a containerized deployment.*

In `api/src/diagnose.py`, `reference_times.json` is opened and parsed at module import time:

```python
_REF_PATH = Path(__file__).parent.parent / "reference_times.json"
with open(_REF_PATH) as _f:
    REFERENCE_TIMES: dict = json.load(_f)
```

This is correct for a containerised deployment for two reasons. First, file I/O and JSON parsing on every request would add latency in proportion to request volume which is not needed since the file never changes during the container's lifetime. Second, it remains consistent. By loading once at startup, this ensures that every request within the same container instance uses an identical copy of the reference data. 

---

## 3.2 Containerization And Deployment

### Question 1
*Walk through your `Dockerfile`. Justify the base image and explain the purpose of each key instruction.*

```dockerfile
FROM python:3.13-slim
```
`python:3.13-slim` is used because the project's `pyproject.toml` requires `python >= 3.13` (union type syntax `float | None` used in `app.py` and `diagnose.py` requires 3.10+, and `3.13-slim` keeps the image small by excluding unnecessary system packages).

```dockerfile
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
```
Copies the `uv` binary from its official image into the container. `uv` is used instead of `pip` because it resolves and installs dependencies significantly faster, reducing build time.

```dockerfile
WORKDIR /app
RUN uv pip install --system fastapi uvicorn
```
Sets the working directory and installs only the two runtime dependencies. `pandas` and `pyarrow` are intentionally excluded — they are only needed by the compute scripts (`compute_reference_times.py`, `generate_validation_set.py`), not by the API itself, keeping the image lean.

```dockerfile
COPY src/ ./src/
COPY reference_times.json .
```
Copies the application source code and bundles `reference_times.json` inside the image. Bundling the reference data ensures it is loaded at startup from the local filesystem with no external dependency on S3 or any network call.

```dockerfile
WORKDIR /app/src
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```
Changes into `src/` so Python can resolve the `diagnose` import in `app.py` without additional `PYTHONPATH` configuration. `EXPOSE 8000` documents the port. The `CMD` starts Uvicorn bound to all interfaces so the container is reachable from outside.

### Question 2
*Besides ECS Fargate, name two other AWS compute options you could have used to deploy this API. For each, list one advantage and one disadvantage compared to the Fargate approach you implemented.*

**AWS Lambda (with Function URL or API Gateway)**
- Advantage: zero infrastructure management and true pay-per-request pricing — cost is effectively zero when the API is idle, whereas our Fargate task accrues cost continuously while running.
- Disadvantage: Each invocation of Lambda starts with no memory from the previous one. For this API, this is a negative as the reference times are loaded at evry start rather than being saved in the memory. Contrarily, Fargate keeps the container running so that data stays loaded between different requests.

**AWS EC2 (e.g. a t3.micro running Uvicorn directly)**
- Advantage: full control over the runtime environment, persistent storage, and the ability to run multiple services on the same instance — potentially cheaper for sustained high-traffic workloads where a single instance is always saturated.
- Disadvantage: requires manual OS patching, capacity planning, and instance management. Fargate abstracts all of this — we only define CPU/memory and the container image, with no SSH access or AMI updates to maintain.

---

## 3.3 Testing And Extensibility

### Question 1
*Why does the exam ask you to test `diagnose()` as a pure function instead of through HTTP requests to the running API?*

Testing `diagnose()` directly has three concrete advantages over HTTP-based testing. First is speed. The 34 tests complete in 0.02 seconds with no server startup overhead, an HTTP-based suite would require spinning up Uvicorn, binding a port, and making network calls for each test. Second, isolation: a pure function test has no dependency on FastAPI, Uvicorn, network stack, or port availability. A failure would point to logic in `diagnose.py`, not to a misconfigured server or a port conflict in CI. Last, debuggability: calling `diagnose()` directly allows passing arbitrary Python dicts and inspecting the return value as a plain dict, without serialising/deserialising JSON or parsing HTTP responses. 

### Question 2
*A new die matrix `6001` enters production. Walk through every change needed — data files, code, tests, and redeployment — to support it.*

1. **Data**: Once production pieces for matrix `6001` are collected into `pieces.parquet`, re-run `api/compute_reference_times.py`. This recomputes medians for all matrices including `6001` and overwrites `api/reference_times.json` with the updated values.

2. **Code**: No changes to `diagnose.py` logic are needed — the delay-detection rule is matrix-agnostic. The only implicit change is that `REFERENCE_TIMES` (loaded from the updated `reference_times.json`) will now contain a `"6001"` key, which means `diagnose()` will accept `die_matrix=6001` without raising a `ValueError`.

3. **Tests**: Add `6001` to the `MATRICES` list in `test_diagnose.py`. The six parametrized test functions (`test_all_ok`, `test_furnace_to_2nd_strike_penalized`, etc.) will automatically generate 6 new test cases for matrix `6001` — no new test functions needed. Also add a `P011` row to `validation_pieces.csv` (all-OK on `6001`), regenerate `validation_expected.json` by re-running `generate_validation_set.py`, and the golden test will pick it up automatically.

4. **Redeployment**: Rebuild the Docker image (`docker build --platform linux/amd64 --provenance=false -t diagnose-api .` from `api/`), push the new image to ECR (`docker push 556455481958.dkr.ecr.eu-west-1.amazonaws.com/vaultech-diagnostics-api:latest`), then register a new ECS task definition revision and stop/restart the Fargate task so it pulls the updated image.
