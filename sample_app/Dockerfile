FROM python:3.12
WORKDIR /app
COPY requiremedddnts.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
 
