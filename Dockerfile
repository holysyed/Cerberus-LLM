FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
WORKDIR /opt/cerberus
COPY pyproject.toml README.md LICENSE ./
COPY cerberus/ ./cerberus/
RUN python -m pip install --no-cache-dir ".[dashboard]" \
    && groupadd --gid 10001 cerberus \
    && useradd --uid 10001 --gid cerberus --create-home cerberus \
    && install -d -o cerberus -g cerberus /data
COPY dashboard/ ./dashboard/
COPY .streamlit/ ./.streamlit/
COPY examples/demo.py ./examples/demo.py
ENV CERBERUS_DATABASE_URL=sqlite:////data/cerberus.sqlite3
USER cerberus
EXPOSE 8000 8501
CMD ["python", "-m", "uvicorn", "cerberus.gateway.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
