# Multi-notebook serving image for the AIF-controller traffic experiments.
#
# Builds the aif_traffic package + its dependencies (marimo, matplotlib,
# numpy, pandas, jax) into a slim Python 3.11 image. The default command
# serves every notebook under notebooks/ at http://0.0.0.0:2718 via
# `marimo run`, headless, no token.
#
#   docker build -t aif-controller .
#   docker run --rm -p 2718:2718 aif-controller
#
# For sharing with collaborators, point a cloudflared tunnel at port 2718
# on the host. Cloudflared lives outside this image.

FROM python:3.11-slim

# uv from the official astral-sh image, no extra apt deps needed.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

# Install dependencies first so that notebook-only edits don't bust the
# dep layer cache.
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

RUN uv sync --frozen --no-dev

COPY notebooks/ ./notebooks/

EXPOSE 2718

CMD ["uv", "run", "marimo", "run", "notebooks", \
     "--host", "0.0.0.0", "--port", "2718", "--headless"]
