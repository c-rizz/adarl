import time
import flask
from flask import Flask, render_template, redirect, request, url_for, Response
# import flask_login
from pathlib import Path
import numpy as np
import cv2

from lr_gym.utils.dbg.web_video_streamer import VideoStreamerSubscriber

# a nice reference can be found at : https://blog.miguelgrinberg.com/post/flask-video-streaming-revisited

# start with: gunicorn --bind 0.0.0.0:9422 lr_gym.utils.dbg.web_video_streamer_app:app


app = Flask(__name__)
app.secret_key = 'ihavenoideawhatthisis-afcsdocvbsdvhboyre934nhauyhnozf834wym'

def generate(topic):
  last_img_t = -1
  max_rate = 30
  while True:
    t0 = time.monotonic()
    images = pipe_server.get_images()
    if len(images)>0:
      t, topic, frame = images[topic]
    else:
      t, topic, frame = last_img_t,"",None
    if t == last_img_t: # if the image hasn't changed
       time.sleep(0.04)
    else:
      ret, buffer = cv2.imencode('.jpg', frame)
      frame = buffer.tobytes()
      yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
    time.sleep(max(1/max_rate - (time.monotonic()-t0), 0))

  

@app.route('/')
def index():
   newline  = "\n"
   return f'''
<!DOCTYPE html>
<html>
<head>
  <meta http-equiv="refresh" content="10" />
  <title>Video Streams</title>
</head>  
<body>
 <h1>Available streams</h1>
 <ul>
  {newline.join("<li><a href="+topic+">"+topic+"</a></li>" for topic in pipe_server.get_images().keys())}
</ul> 
</body>
</html>'''


@app.route("/<topic>/stream")
def video_feed(topic):
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(topic),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route("/<topic>")
def video_page(topic):
   return f'''
<html>
  <head>
    <title>{topic}</title>
    <style>
      img {{
        height: 90%;
      }}
    </style>
  </head>
  <body>
    <h1>{topic}</h1>
    <img src="/{topic}/stream">
  </body>
</html>'''


with app.app_context():
  pipe_server = VideoStreamerSubscriber()
   

if __name__ == '__main__':
  app.run(debug=False, host="0.0.0.0")
