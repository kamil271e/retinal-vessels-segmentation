FROM python:3.11.4
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN chmod 777 efnet_hotfix.sh
RUN ./efnet_hotfix.sh
RUN rm efnet_hotfix.sh

CMD ["python", "full_pipe.py"]
