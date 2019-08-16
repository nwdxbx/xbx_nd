#import -----------------
import os.path as osp
import sys,os

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = os.path.dirname(__file__)



# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, '..',  'lib')
add_path(lib_path)

#export -----------------
from face_detect import FaceDetection


