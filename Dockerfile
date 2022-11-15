FROM pytorch/pytorch:latest

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

WORKDIR .
COPY FDST-HR.pth ./FDST-HR.pth

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY datasets ./datasets
COPY misc ./misc
COPY model ./model
COPY templates ./templates
COPY config.py ./config.py

RUN sed -i "s/self._ydl_info\['like_count'\]/0/g" /opt/conda/lib/python3.9/site-packages/pafy/backend_youtube_dl.py
RUN sed -i "s/self._ydl_info\['dislike_count'\]/0/g" /opt/conda/lib/python3.9/site-packages/pafy/backend_youtube_dl.py

COPY app.py ./app.py
EXPOSE 5000
CMD [ "python", "app.py"]