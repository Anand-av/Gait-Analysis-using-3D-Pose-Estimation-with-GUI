#   Gait-Analysis-using-3D-Pose-Estimation-with-GUI

#### Dependencies
    Python3
    Tensorflow 1.3.6, Tensorflow Graphs
    Opencv3, python3-tk, protobuf
    Build c++ library 
    cd tf_pose/pafprocess
    swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace
    
#### Requiremnets
    argparse
    matplotlib
    scipy
    tqdm
    requests
    fire
    dill
    git+https://github.com/ppwwyyxx/tensorpack.git
    PyQt 4.7+, PySide, or PyQt5
    NumPy
    For 3D graphics: pyopengl and qt-opengl

#### ESTIMATE THE 3D POSE IN REALTIME
1.Scenario where a person walks Infront of the camera. The person may be a Normal/Abnormal person. 

2.Post this scenario, Extraction of coordinates values. The coordinate values are collected in a txt file in the form of a list.

3.Then there are three gait measures extracted in this work: Step length, Step width and Gait speed.

4.The calculation is computed for Mean, standard deviation, minimum and maximum values of the gait measures.

5.Post this we are evaluating the test records with the trained model. 

6.Logistic Regression and Naive Bayes algorithms are used for evaluation and finally accuracy is obtained.

7.There is a small GUI that is being built in order to display the main highlights of the process and the results. (There is a separate doc available on the Interface and working of the GUI)

#### TEST INTERFERENCE, FROM IMAGES, WE ESTIMATE THE 3D POSE 
1.We upload a Jpg image of a person.

2.Once we execute the run file using the below command:
Python run.py --image==exampleimg.jpg

3.We get all the 2D key points and those are connected with best possible straight lines. 

4.We also get the heat map, Vector map-x and Vector map-y along.

5.Finally using these 2D key points we estimate the 3D pose

#### Main repository used:https://github.com/ZheC/tf-pose-estimation

#### Repository used for 3D plotting:https://github.com/pyqtgraph/pyqtgraph
