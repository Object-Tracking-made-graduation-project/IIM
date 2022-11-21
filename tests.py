import os

import cv2
import numpy as np
import pytest
from imutils.video import FileVideoStream

#os.environ["PAFY_BACKEND"] = "internal"  # noqa E402

import pafy
import youtube_dl

URL = "https://www.youtube.com/embed/2wqpy036z24"
URL_MODEL = "https://www.youtube.com/watch?v=2wqpy036z24"




def test_can_use_pafy(url=URL_MODEL, stream_num=-1):
    print(pafy.__version__)
    #url = URL
    pafy_obj = pafy.new(url)
    play = pafy_obj.streams[stream_num]
    assert play is not None
    return cv2.VideoCapture(play.url)


def test_can_use_file_video_stream():
    def get_video_stream_url(url, stream_num) -> str:
        pafy_obj = pafy.new(url)
        play = pafy_obj.streams[stream_num]
        assert play is not None
        return play.url
    fvs = FileVideoStream(get_video_stream_url(URL_MODEL, -3)).start()
    assert fvs is not None
    frame = fvs.read()
    assert isinstance(frame, np.ndarray)



