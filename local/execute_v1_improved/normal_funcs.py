import deepeye
import numpy as np
import cv2

def MakeNPImage(filename):
	imgFile = open(filename, "rb")
	img_data = imgFile.read()
	npimg = np.frombuffer(img_data, dtype=np.uint8)
	return npimg

def MakeDeepEye(model_path):
	return deepeye.DeepEye(model_path)

def MakeNetwork(eyetracker):
    return deepeye.Network('Deep_eye', eyetracker.input_reshaped_casted, 1, is_training=False, reuse=False, deep=2, layers=16)

def SetNetwork(eyetracker, network_model):
    eyetracker.setNetwork(network_model)

def MakeProbabilityMask(eyetracker, frame_gray):
	return eyetracker.run(frame_gray)

def MakeBlobLocations(prob_mask):
        factor = prob_mask.size / (288.0 * 384.0)

        params = cv2.SimpleBlobDetector_Params()

        params.filterByArea = True
        params.minArea = 100 * factor
        params.maxArea = 25000 * factor

        params.filterByConvexity = True
        params.minConvexity = 0.1

        detector = cv2.SimpleBlobDetector_create(params)

        found_blob = False
        prob = 0.3 # 0.5
        raw_img = prob_mask.copy()
        while found_blob == False:
            image = raw_img.copy()
            image[image < prob] = 0
            image[image > prob] = 1
            image = (image * 255).astype('uint8')
            image = 255 - image
            keypoints = detector.detect(image)

            if len(keypoints) > 0:

                blob_sizes = []
                for k in keypoints:
                    blob_sizes.append(k.size)
                detection = np.argmax(np.asarray(blob_sizes))
                out_coordenate = [int(keypoints[detection].pt[0]), int(keypoints[detection].pt[1])]
                found_blob = True
            else:
                out_coordenate = [0, 0]
                found_blob = False
                prob += -0.05
                if prob < 0.05:
                    found_blob = True

        return (out_coordenate, image)
        
def MakeFrameAndGrayFrame(npimg):
    frame = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame, frame_gray

def ProcessSingleImage(frame, frame_gray, blob_locs):
        #img_data = base64.b64decode(image_encoded_data)
        #npimg = np.frombuffer(img_data, dtype=np.uint8)
        # PUPIL!!!
        (coords, bw_img) = blob_locs
        # contours, _ = cv2.findContours(bw_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        _, contours, _ = cv2.findContours(bw_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_poly = [None] * len(contours)
        boundRect = [None] * len(contours)
        centers = [[]] * len(contours)
        pupilRadiuses = [0] * len(contours)

        for i, c in enumerate(contours):
            contours_poly[i] = cv2.approxPolyDP(c, 3, True)
            boundRect[i] = cv2.boundingRect(contours_poly[i])
            centers[i], pupilRadiuses[i] = cv2.minEnclosingCircle(contours_poly[i])

        pupilX = pupilY = pupilR = float('-inf')
        for i in range(len(contours)):
            if 2 * pupilRadiuses[i] > bw_img.shape[1] / 2:
                # filter big diameter
                continue
            if pupilRadiuses[i] > pupilR:
                pupilX = int(centers[i][0])
                pupilY = int(centers[i][1])
                pupilR = int(pupilRadiuses[i])
        pupilX = -1 if pupilX == float('-inf') else pupilX
        pupilY = -1 if pupilY == float('-inf') else pupilY
        pupilR = -1 if pupilR == float('-inf') else pupilR

        # IRIS!!!!!
        irisCannyThreshold = 60
        irisAccumulatorThreshold = 40
        irisRadMin = 3 * int(frame_gray.shape[1] / 40)
        irisRadMax = 10 * int(frame_gray.shape[1] / 40)
        irisCircles = cv2.HoughCircles(cv2.cvtColor(cv2.medianBlur(frame,5), cv2.COLOR_BGR2GRAY), cv2.HOUGH_GRADIENT, 1, 1,
                                  param1=irisCannyThreshold, param2=irisAccumulatorThreshold, minRadius=irisRadMin, maxRadius=irisRadMax)
        irisCircles = np.uint16(np.around(irisCircles))

        irisX = irisY = irisR = -1
        for candidate in irisCircles[0, :]:
            tempIrisX, tempIrisY, tempIrisR = candidate[0], candidate[1],candidate[2]
            curDist = np.sqrt(np.sum(np.square(np.array([tempIrisX, tempIrisY]) - np.array([pupilX, pupilY]))))
            if curDist + pupilR >= tempIrisR:
                # wrong location for iris
                continue
            irisX, irisY, irisR = tempIrisX, tempIrisY, tempIrisR
            break

        height = frame_gray.shape[0]
        width = frame_gray.shape[1]
        return {
            "pupilIrisRatio" : float(pupilR) / float(irisR),
            "imgRatio": float(height) / float(width),
            "pupil": {
                "r" : float(pupilR) / float(width),
                "x" : float(pupilX) / float(width),
                "y": float(pupilY) / float(height),
            },
            "iris": {
                "r": float(irisR) / float(width),
                "x": float(irisX) / float(width),
                "y": float(irisY) / float(height),
            }
        }