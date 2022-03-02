import numpy as np
import cv2

@profile
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

if __name__ == '__main__':
	probability_mask = np.load("func2_A2_probability_mask.npy")
	blob_locs = MakeBlobLocations(probability_mask)
	np.save("func2_A3_blob_locations", blob_locs)