FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY perfmath perfmath
COPY repro.py test_perfmath.py PROMPT.md METADATA.json ./

CMD ["pytest", "-q"]