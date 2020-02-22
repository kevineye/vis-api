Run [YOLOv3](https://pjreddie.com/darknet/yolo/) as a web service. Detects 80+ types of objects from the [CoCo dataset](http://cocodataset.org/) and returns an annotated PNG and metadata in HTTP headers.

    $ curl -X POST -D - -F image=@dog.jpg http://localhost:5000/api/detect -o dog-annotated.png
    HTTP/1.0 200 OK
    Content-Type: image/png
    Content-Length: 683268
    X-Detect-Count: 2
    X-Detect-1-Label: dog
    X-Detect-1-Confidence: 0.8278917074203491
    X-Detect-1-Bounds: [124, 218, 257, 299]
    X-Detect-2-Label: car
    X-Detect-2-Confidence: 0.7426347732543945
    X-Detect-2-Bounds: [466, 82, 219, 89]

---

Based on code and an excellent write up at https://medium.com/analytics-vidhya/object-detection-using-yolo-v3-and-deploying-it-on-docker-and-minikube-c1192e81ae7a
