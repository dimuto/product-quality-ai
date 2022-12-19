# install libraries
FROM python:3.7
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# install cv2 dependencies
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

# run python script
COPY . . 
WORKDIR "src"
CMD ["python3", "-m", "app", "--host=0.0.0.0"]