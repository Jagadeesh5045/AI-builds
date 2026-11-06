# Authored artefact. There is no Docker in the build environment used for this
# project, so this image was NOT built or run. It is statically reviewed only.
FROM python:3.12-slim

WORKDIR /srv/app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ingestion ./ingestion
COPY analysis ./analysis
COPY storage ./storage
COPY app ./app
COPY data ./data

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
