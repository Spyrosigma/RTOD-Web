import cv2
from flask import Flask, Response, render_template
import numpy as np
from ultralytics import YOLO
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = YOLO("yolov8n.pt")

def image():
    pass

def WebcamDetection():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            results = model.predict(frame)
            for r in results:
                im_array = r.plot()
            ret, buffer = cv2.imencode(".jpg", im_array)
            im_array = buffer.tobytes()
            yield (
                b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + im_array + b"\r\n"
            )

def video():    
    # def resize_video(input_video_path, output_video_path,
    #                 new_width, new_height):
    #     cap = cv2.VideoCapture(input_video_path)
    #     if not cap.isOpened():
    #         print("Error opening video:", input_video_path)
    #         return
        
    #     original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #     fps = cap.get(cv2.CAP_PROP_FPS)
    #     fourcc = cv2.VideoWriter_fourcc(*"XVID") #codec

    #     scale_ratio = min(new_width / original_width, new_height / original_height)
    #     new_width = int(original_width * scale_ratio)
    #     new_height = int(original_height * scale_ratio)

    #     writer = cv2.VideoWriter(output_video_path, fourcc, fps, (new_width, new_height))

    #     while True:
    #         success, frame = cap.read()
    #         if not success:
    #             print("Error reading frame.")
    #             break

    #         resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    #         writer.write(resized_frame)

    #         cv2.imshow('Resized Video', resized_frame)

    #         if cv2.waitKey(1) & 0xFF == ord("q") or not success:
    #             break
    #     cap.release()
    #     writer.release()
    pass

#-------------------Routes-----------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/image")
def image():
    return render_template("image.html")


@app.route("/webcam")
def webcam():
    return Response(WebcamDetection(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video")
def video():
    return render_template("video.html")



if __name__ == "__main__":
    app.run(debug=True)
