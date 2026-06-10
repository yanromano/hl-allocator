FROM python:3.14-slim AS base
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

# Install runtime deps first (layer cache).
COPY pyproject.toml ./
RUN pip install --no-cache-dir .

# App code.
COPY automation/ ./automation/

# Reinstall the dist so console scripts see the package (deps layer above stays cached).
RUN pip install --no-cache-dir --no-deps .

# Non-root.
RUN useradd --create-home --uid 10001 app && chown -R app:app /app
USER app

EXPOSE 8080
# /data is the Railway volume: state files AND the operator-provided automation.yaml live there.
CMD ["python", "-m", "automation.scripts.run_daemon", "--config", "/data/automation.yaml"]
