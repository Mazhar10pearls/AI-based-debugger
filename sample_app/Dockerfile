FROM python:3.1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirementsss.txt
COPY . .
EXPOSE 5000
CMD ["python", "apsddsp.py"]
