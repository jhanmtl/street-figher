from flask import Flask, Response
from cv_camera import Camera
import argparse

ap=argparse.ArgumentParser()
ap.add_argument('reduction_factor',type=int)
args=ap.parse_args()
rf=args.reduction_factor


app = Flask(__name__)

@app.route('/')
def index():
    cam=Camera(rf)
    return Response(gen(cam),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)