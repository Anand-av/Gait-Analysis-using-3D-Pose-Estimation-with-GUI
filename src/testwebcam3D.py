"""
This serve as our base openGL class.
"""

import numpy as np
import pyqtgraph.opengl as gl
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import sys
import cv2
import time
import os
import csv

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh
from lifting.prob_model import Prob3dPose
import common



class Terrain(object):
    def __init__(self):
        """
        Initialize the graphics window and mesh surface
        """

        # setup the view window
        self.app = QtGui.QApplication(sys.argv)
        self.window = gl.GLViewWidget()
        self.window.setWindowTitle('Terrain')
        self.window.setGeometry(0, 110, 1920, 1080)
        self.window.setCameraPosition(distance=30, elevation=12)
        self.window.show()

        gx = gl.GLGridItem()
        gy = gl.GLGridItem()
        gz = gl.GLGridItem()
        gx.rotate(90, 0, 1, 0)
        gy.rotate(90, 1, 0, 0)
        gx.translate(-10, 0, 0)
        gy.translate(0, -10, 0)
        gz.translate(0, 0, -10)
        self.window.addItem(gx)
        self.window.addItem(gy)
        self.window.addItem(gz)

        model = 'mobilenet_thin'
        #model = '432x368'
        camera = 0
        self.lines = {}
        self.connection = [
            [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6],
            [0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12],
            [12, 13], [8, 14], [14, 15], [15, 16]
        ]
        
        resolution = '432x368'
        w, h = model_wh(resolution)
        self.e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))
        self.cam = cv2.VideoCapture(camera)
        ret_val, image = self.cam.read()        
        #print('ret_val', ret_val)
        self.poseLifting = Prob3dPose('./src/lifting/models/prob_model_params.mat')
        keypoints = self.mesh(image)
        print('keypoints', keypoints)

        self.points = gl.GLScatterPlotItem(
            pos=keypoints,
            color=pg.glColor((0, 255, 0)),
            size=15
        )
        #print(keypoints)
        self.window.addItem(self.points)
        
        for n, pts in enumerate(self.connection):
            self.lines[n] = gl.GLLinePlotItem(
                pos=np.array([keypoints[p] for p in pts]),
                color=pg.glColor((0, 0, 255)),
                width=3,
                antialias=True
            )
            self.window.addItem(self.lines[n])


    def mesh(self, image):
        image_h, image_w = image.shape[:2]
        width = 640
        height = 480
        pose_2d_mpiis = []
        visibilities = []

        humans = self.e.inference(image, scales=[None])
        #print('humans3', humans)
        
            
        for human in humans:
            pose_2d_mpii, visibility = common.MPIIPart.from_coco(human)
            pose_2d_mpiis.append(
                [(int(x * width + 0.5), int(y * height + 0.5)) for x, y in pose_2d_mpii]
            )
            visibilities.append(visibility)

        pose_2d_mpiis = np.array(pose_2d_mpiis)
        #print('pose_2d_mpiis', pose_2d_mpiis)
        visibilities = np.array(visibilities)
        transformed_pose2d, weights = self.poseLifting.transform_joints(pose_2d_mpiis, visibilities)
        #print('transformed_pose2d', transformed_pose2d)
        pose_3d = self.poseLifting.compute_3d(transformed_pose2d, weights)
        #print(type(pose_3d))
        lis_3d = pose_3d.tolist()
        print('list_3d', lis_3d)
        #fil_pose = open('C:\\Users\\carti\\Desktop\\tf-pose\\src\\pose_3d.txt', 'w')
        #fil_pose.write(lis_3d)
        #fil_pose.close()
        with open('C:\\Users\\carti\\Desktop\\tf-pose\\src\\pose_3d.txt', 'a') as f:
            for item in lis_3d:
                f.write("%s\n" % item)
        #np.savetxt('C:\\Users\\carti\\Desktop\\tf-pose\\src\\pose_3da.txt', pose_3d)
        
        #with open('C:\\Users\\carti\\Desktop\\tf-pose\\src\\pose_3da.csv', 'a') as csvfile:
        #    writer = csv.writer(csvfile, delimiter=",")
        #    writer.writerow(lis_3d)
        
        #print(lis_3d[0])
        #print('pose_3d', pose_3d.transpose())
        #pose_chk1 = pose_3d.transpose()
        #pose_chk = pose_chk1 / 80
        #print('pose_3d', pose_chk)
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
            
            for n, pts in enumerate(self.connection):
                self.lines[n].setData(
                    pos=np.array([keypoints[p] for p in pts])
                )

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
        #Timer(5, self.exitfunc).start()        
        #t = threading.Thread(target=self.listen)
        #t.daemon = True
        #t.start()
        #time.sleep(3)
        

if __name__ == '__main__':
    os.chdir('..')    
    t = Terrain()
    t.animation()        
    
