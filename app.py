import os

#os.environ["PAFY_BACKEND"] = "internal"  # noqa E402

import cv2
import pafy
from flask import Flask, render_template, Response
from imutils.video import FileVideoStream

from misc.image import get_points_on_image
from misc.params import InferenceParams, ModelParams, ImageParams
from model.utils import load_model

app = Flask(__name__)

URL = "https://www.youtube.com/embed/2wqpy036z24"
URL_MODEL = "https://www.youtube.com/watch?v=2wqpy036z24"

params = InferenceParams(model_params=ModelParams())  # todo: Здесь должны быть параметры модели

model = load_model(params.model_params)
model.eval()


def get_video_obj(url, stream_num=-1):
    """
    Creates a new video streaming object to extract video frame by frame to make prediction on.
    :return: opencv2 video capture object, with lowest quality frame available for video.
    """
    pafy_obj = pafy.new(url)
    play = pafy_obj.streams[stream_num]
    assert play is not None
    return cv2.VideoCapture(play.url)


def get_video_stream_url(url, stream_num) -> str:
    pafy_obj = pafy.new(url)
    play = pafy_obj.streams[stream_num]
    assert play is not None
    return play.url


def gen_frames():  # generate frame by frame from camera
    video_obj = get_video_obj(URL_MODEL, -3)  # use 0 for web camera
    while video_obj.isOpened():
        success, frame = video_obj.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


def gen_det_frames():
    video_obj = get_video_obj(URL_MODEL, -3)  # поток -3 имеет разрешение меньше , чем -1
    while video_obj.isOpened():
        ret, frame = video_obj.read()
        points = get_points_on_image(frame, model)
        size = 5
        for x, y in points:
            frame = cv2.circle(frame, (int(x), int(y)), size, (0, 255, 255), -1)
        det_ret, det_buffer = cv2.imencode('.jpg', frame)
        out_frame = det_buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + out_frame + b'\r\n')  # concat frame one by one and show result


def gen_det_frames_imutils():
    fvs = FileVideoStream(get_video_stream_url(URL_MODEL, -3)).start()

    while fvs.more():
        frame = fvs.read()
        points = [] # get_points_on_image(frame, model)
        size = 5
        for x, y in points:
            frame = cv2.circle(frame, (int(x), int(y)), size, (0, 255, 255), -1)
        det_ret, det_buffer = cv2.imencode('.jpg', frame)
        out_frame = det_buffer.tobytes()
        fvs.update()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + out_frame + b'\r\n')  # concat frame one by one and show result


@app.route('/raw_video_feed')
def raw_video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/det_video_feed')
def det_video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    generator = gen_det_frames() # gen_det_frames_imutils()
    return Response(generator, mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    print("Flask app started.")
    app.run(host='0.0.0.0')
