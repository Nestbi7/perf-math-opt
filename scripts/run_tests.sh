mkdir -p scripts
cat > scripts/run_tests.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

rm -f .pytest_report.json

pytest -q --json-report --json-report-file=.pytest_report.json

python - <<'PY'
import json

with open(".pytest_report.json", "r", encoding="utf-8") as f:
    report = json.load(f)

tests = []
for t in report.get("tests", []):
    nodeid = t.get("nodeid", "")
    outcome = (t.get("outcome") or "").lower()
    if outcome == "passed":
        status = "PASSED"
    elif outcome == "failed":
        status = "FAIL"
    elif outcome == "skipped":
        status = "SKIPPED"
    else:
        status = "ERROR"
    tests.append({"name": nodeid, "status": status})

print(json.dumps({"tests": tests}, indent=2))
PY
SH

chmod +x scripts/run_tests.sh