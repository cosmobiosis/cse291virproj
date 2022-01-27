import ffmpeg
import numpy as np

fileName = 'inputs/concated.mp4'
probe = ffmpeg.probe(fileName)
video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
width = int(video_info['width'])
height = int(video_info['height'])

out, _ = (
    ffmpeg
    .input(fileName)
    .output('pipe:', format='rawvideo', pix_fmt='rgb24')
    .run(capture_stdout=True)
)
video = (
    np
    .frombuffer(out, np.uint8)
    .reshape([-1, height, width, 3])
)

np.save("inputs/concated", video)