FROM python:3.11

ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]