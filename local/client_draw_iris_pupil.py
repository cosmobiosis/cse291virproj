import json
import cv2

data = {'pupilIrisRatio': 0.35294117647058826, 'imgRatio': 0.9510086455331412, 'pupil': {'r': 0.069164265129683, 'x': 0.49279538904899134, 'y': 0.4212121212121212}, 'iris': {'r': 0.19596541786743515, 'x': 0.5072046109510087, 'y': 0.41818181818181815}}
path = 'samples/eye6.jpg'
# Reading an image in default mode
image = cv2.imread(path)
# Window name in which image is displayed
window_name = 'pupil iris'
# height, width, number of channels in image
height = image.shape[0]
width = image.shape[1]

token = "pupil"
# Center coordinates
center_coordinates = (int(data[token]["x"] * width), int(data[token]["y"] * height))
# Radius of circle
radius = int(width * data[token]["r"])
# Blue color in BGR
color = (0, 0, 255)
# Line thickness of 2 px
thickness = 2
# Using cv2.circle() method
# Draw a circle with blue line borders of thickness of 2 px
image = cv2.circle(image, center_coordinates, radius, color, thickness)

token = "iris"
# Center coordinates
center_coordinates = (int(data[token]["x"] * width), int(data[token]["y"] * height))
# Radius of circle
radius = int(width * data[token]["r"])
# Blue color in BGR
color = (0, 255, 0)
# Line thickness of 2 px
thickness = 2
# Using cv2.circle() method
# Draw a circle with blue line borders of thickness of 2 px
image = cv2.circle(image, center_coordinates, radius, color, thickness)

# Displaying the image
cv2.imwrite("client_draw_output.jpg", image)