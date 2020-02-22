import cv2
import json
import numpy as np
import time

from flask import Flask, request, Response
from io import BytesIO
from PIL import Image


def get_labels():
    return open("/dataset/coco.names").read().strip().split("\n")

def get_colors(labels):
    np.random.seed(42)
    return np.random.randint(0, 255, size=(len(labels), 3),dtype="uint8")

def load_model():
    print("[INFO] loading YOLO from disk...")
    return cv2.dnn.readNetFromDarknet("/dataset/yolov3.cfg", "/dataset/yolov3.weights")

def image_to_byte_array(image:Image):
  imgByteArr = BytesIO()
  image.save(imgByteArr, format='PNG')
  return imgByteArr.getvalue()

def get_prediction(image,net,labels,colors):
    (h, w) = image.shape[:2]

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    print(layerOutputs)
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            # print(scores)
            classID = np.argmax(scores)
            # print(classID)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confthres:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confthres, nmsthres)

    res = {
        "detections": []
    }

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in colors[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
            res["detections"].append({
                "bounds": boxes[i],
                "label": labels[classIDs[i]],
                "confidence": confidences[i],
            })
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)

    res["image"] = image
    return res


confthres = 0.3
nmsthres = 0.1

labels = get_labels()
colors = get_colors(labels)
nets = load_model()

app = Flask(__name__)

@app.route('/api/detect', methods=['POST'])
def main():
    img = Image.open(BytesIO(request.files["image"].read()))
    image = cv2.cvtColor(np.array(img).copy(), cv2.COLOR_BGR2RGB)
    res = get_prediction(image, nets, labels, colors)
    img_encoded = image_to_byte_array(Image.fromarray(cv2.cvtColor(res["image"], cv2.COLOR_BGR2RGB)))
    resp = Response(response=img_encoded, status=200, mimetype="image/png")
    resp.headers["X-Detect-Count"] = len(res["detections"])
    for i, d in enumerate(res["detections"], start=1):
        resp.headers["X-Detect-" + str(i) + "-Label"] = d["label"]
        resp.headers["X-Detect-" + str(i) + "-Confidence"] = d["confidence"]
        resp.headers["X-Detect-" + str(i) + "-Bounds"] = json.dumps(d["bounds"])
    return resp



if __name__ == '__main__':
    app.run(host='0.0.0.0')
