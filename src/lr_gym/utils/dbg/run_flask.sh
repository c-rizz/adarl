#!/bin/bash

cd $(dirname $0)
flask --app web_video_streamer_app.py run -p 9422 --host=0.0.0.0
