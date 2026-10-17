# Two stages: build a virtualenv and train the models, then copy only the venv and the model
# files into a clean slim image. Debian codename pinned (plain -slim tags move to new releases).
FROM python:3.12-slim-bookworm AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install . && canary train --out /opt/models

FROM python:3.12-slim-bookworm
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/models \
    MODEL_VERSION=v1
RUN groupadd --system --gid 10001 app && useradd --system --uid 10001 --gid app --no-create-home app
COPY --from=build /opt/venv /opt/venv
COPY --from=build /opt/models /app/models
WORKDIR /app
USER 10001
EXPOSE 8000
CMD ["canary", "serve", "--port", "8000"]
