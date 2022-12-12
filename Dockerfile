FROM python:3.7
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY . . 
CMD ["python", "src/app.py"]