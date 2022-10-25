# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import tensorflow as tf

def import_openpose():
    try:
        sys.path.append('../../python')
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e
    return op

try:
    op = import_openpose()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "../../../models/"

    # Starting OpenPose 
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()


    videos_path = tf.io.gfile.glob(
        '/home/alvaro/Documents/AUTSL_VIDEO_DATA/test/test/*.mp4')

    for video in videos_path:
        cap = cv2.VideoCapture(video)
        while True:
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process Image
            datum = op.Datum()
            imageToProcess = frame
            datum.cvInputData = imageToProcess
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))

            # Display Image
            print("Body keypoints: \n" + str(datum.poseKeypoints))
            cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
            cv2.waitKey(0)
except Exception as e:
    print(e)
    sys.exit(-1)
