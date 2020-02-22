FROM python:3.8-slim

RUN apt-get update \
 && apt-get install -q -y curl libglib2.0-0 libsm6 libxrender-dev libxext6 \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir /dataset

# switch to these for full yolov3 dataset
#RUN curl -LfsSO https://pjreddie.com/media/files/yolov3.weights
#RUN curl -LfsSO https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg

# use yolov3-tiny
RUN curl -LfsS -o /dataset/yolov3.weights https://pjreddie.com/media/files/yolov3-tiny.weights
RUN curl -LfsS -o /dataset/yolov3.cfg https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg

RUN curl -LfsS -o /dataset/coco.names https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names

WORKDIR /usr/src/app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . /usr/src/app

EXPOSE 5000

ENTRYPOINT [ "python3" ]
CMD [ "app.py" ]
