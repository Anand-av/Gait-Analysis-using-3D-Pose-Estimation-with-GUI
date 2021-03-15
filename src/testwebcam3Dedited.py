"""
This serve as our base openGL class.
"""

import numpy as np
import pyqtgraph.opengl as gl
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import argparse
import sys
import logging
import time
import cv2
import os

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh
from lifting.prob_model import Prob3dPose
import common

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

class Terrain(object):
    def __init__(self):

        #print('path of pose est', os.getcwd())
        parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
        parser.add_argument('--camera', type=int, default=0)
        parser.add_argument('--zoom', type=float, default=1.0)
        parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
        parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
        parser.add_argument('--show-process', type=bool, default=False,
                            help='for debug purpose, if enabled, speed for inference is dropped.')
        args = parser.parse_args()

        logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
        w, h = model_wh(args.resolution)
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
        logger.debug('cam read+')
        cam = cv2.VideoCapture(args.camera)
        ret_val, image = cam.read()
        #print('ret_val', ret_val)..Done
        print('img', image)
        
        while True:
            ret_val, image = cam.read()
            #print('ret_val', ret_val)...Done
            print('img', image)

            logger.debug('image preprocess+')
            if args.zoom < 1.0:
                canvas = np.zeros_like(image)
                img_scaled = cv2.resize(image, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_LINEAR)
                dx = (canvas.shape[1] - img_scaled.shape[1]) // 2
                dy = (canvas.shape[0] - img_scaled.shape[0]) // 2
                canvas[dy:dy + img_scaled.shape[0], dx:dx + img_scaled.shape[1]] = img_scaled
                image = canvas
            elif args.zoom > 1.0:
                img_scaled = cv2.resize(image, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_LINEAR)
                dx = (img_scaled.shape[1] - image.shape[1]) // 2
                dy = (img_scaled.shape[0] - image.shape[0]) // 2
                image = img_scaled[dy:image.shape[0], dx:image.shape[1]]

            print('img1', image)
            logger.debug('image process+')
            humans = e.inference(image)        

            logger.debug('postprocess+')
            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
            print('img2', image)

            logger.debug('show+')
            
            fps_time = 0
            
            cv2.putText(image,
                        "FPS: %f" % (1.0 / (time.time() - fps_time)),
                        (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
            cv2.imshow('tf-pose-estimation result', image)
            fps_time = time.time()
            if cv2.waitKey(1) == 27:
                break
            logger.debug('finished+')

            #cv2.destroyAllWindows()    

            # setup the view window
            app = QtGui.QApplication(sys.argv)
            window = gl.GLViewWidget()
            window.setWindowTitle('Terrain')
            window.setGeometry(0, 110, 1920, 1080)
            window.setCameraPosition(distance=30, elevation=12)
            window.show()

            gx = gl.GLGridItem()
            gy = gl.GLGridItem()
            gz = gl.GLGridItem()
            gx.rotate(90, 0, 1, 0)
            gy.rotate(90, 1, 0, 0)
            gx.translate(-10, 0, 0)
            gy.translate(0, -10, 0)
            gz.translate(0, 0, -10)
            window.addItem(gx)
            window.addItem(gy)
            window.addItem(gz)            
            
            poseLifting = Prob3dPose('./lifting/models/prob_model_params.mat')
            
            #keypoints = self.mesh(image)

            #points = gl.GLScatterPlotItem(
            #    pos=keypoints,
            #    color=pg.glColor((0, 255, 0)),
            #    size=15
            #)
            #window.addItem(points)    
            
    def mesh(self, image):
        image_h, image_w = image.shape[:2]
        width = 640
        height = 480
        pose_2d_mpiis = []
        visibilities = []
        model = 'mobilenet_thin'
        self.e = TfPoseEstimator(get_graph_path(model), target_size=(width, height))
        humans = self.e.inference(image, scales=[None])

        for human in humans:
            pose_2d_mpii, visibility = common.MPIIPart.from_coco(human)
            pose_2d_mpiis.append(
                [(int(x * width + 0.5), int(y * height + 0.5)) for x, y in pose_2d_mpii]
            )
            visibilities.append(visibility)

        pose_2d_mpiis = np.array(pose_2d_mpiis)
        visibilities = np.array(visibilities)
        
        poseLifting = Prob3dPose('./lifting/models/prob_model_params.mat')
        
        transformed_pose2d, weights = poseLifting.transform_joints(pose_2d_mpiis, visibilities)
        pose_3d = self.poseLifting.compute_3d(transformed_pose2d, weights)

        keypoints = pose_3d[0].transpose()

        return keypoints / 80        

    def update(self):
        """
        update the mesh and shift the noise each time
        """
        ret_val, image = self.cam.read()
        try:
            keypoints = self.mesh(image)
        except AssertionError:
            print('body not in image')
        else:
            self.points.setData(pos=keypoints)

    def start(self):
        """
        get the graphics window open and setup
        """
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

    def animation(self, frametime=10):
        """
        calls the update method to run in a loop
        """
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(frametime)
        self.start()
    
if __name__ == '__main__':
    #os.chdir('..')
    #print('path', os.getcwd())
    a_long_time = 5
    time.sleep(a_long_time)
    TIMEOUT = 15
    t = Terrain()
    t.animation()