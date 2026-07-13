ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION} 
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "apssp.py"]
 