import cv2, os
from flask import Flask, Response, render_template, request, redirect, url_for, flash
import numpy as np
from PIL import Image
from ultralytics import YOLO
from flask_cors import CORS
from dotenv import load_dotenv
load_dotenv()

model = YOLO("yolov8n.pt")

app = Flask(__name__)
CORS(app)
app.secret_key = os.getenv('SECRET_KEY')
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MODEL_FOLDER'] = 'static/model/'

IMAGE_ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
def allowed_image_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in IMAGE_ALLOWED_EXTENSIONS

VIDEO_ALLOWED_EXTENSIONS = set(['mp4', 'avi'])
def allowed_video_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in VIDEO_ALLOWED_EXTENSIONS


#-------------------Routes-----------------

@app.route("/")
def home():
    return render_template("index.html")


#------------Image-------------

@app.route('/image')
def image():
    return render_template('image.html')

@app.route('/image', methods=['POST'])
def upload_image():
    file = request.files['file']
    print('File:',file,'\nFileName',file.filename)
    if file and allowed_image_file(file.filename):
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        results = model(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        for r in results:
            im_array = r.plot() 
            im = Image.fromarray(im_array[..., ::-1]) 
            im.save(os.path.join(app.config['MODEL_FOLDER'], filename))
        flash('Image successfully uploaded, Processed and Detected Objects are shown below')
        return render_template('image.html', filename=filename)
    else:
        flash('WRONG Image Type !! Allowed image types are - png, jpg, jpeg')
        return redirect('image')
 
@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='model/' + filename))


#-------------------Webcam-----------------

@app.route("/webcam")
def webcam():
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
    return Response(WebcamDetection(), mimetype="multipart/x-mixed-replace; boundary=frame")


#-------------------Video-----------------

@app.route("/video")
def video():
    return render_template("video.html")

@app.route("/video", methods=["POST"])
def upload_video():
    file = request.files["file"]
    if file and allowed_video_file(file.filename):
        filename = file.filename
        file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
        def save_predicted_video(input_video_path, output_video_path):
            cap = cv2.VideoCapture(input_video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

            while cap.isOpened():
                success, frame = cap.read()
                if success:
                    results = model(frame)
                    annotated_frame = results[0].plot()
                    out.write(annotated_frame)
                else:
                    break
        save_predicted_video(os.path.join(app.config["UPLOAD_FOLDER"], filename), os.path.join(app.config["MODEL_FOLDER"], filename))
        flash("Video successfully uploaded")
        return render_template("video.html", filename=filename)
    else:
        flash("WRONG Video Type !! Allowed video types are - mp4, avi")
        return redirect("video")

@app.route("/display_video/<filename>")
def display_video(filename):
    return redirect(url_for("static", filename="model/" + filename))


