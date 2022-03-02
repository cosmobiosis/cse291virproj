import numpy as np

#mprofile_20770000000000
@profile
def MakeNPImage(filename):
	imgFile = open(filename, "rb")
	img_data = imgFile.read()
	npimg = np.frombuffer(img_data, dtype=np.uint8)
	return npimg

if __name__ == '__main__':
	npimg = MakeNPImage("../samples/eye0.jpg")
	np.save("func1_output", npimg)