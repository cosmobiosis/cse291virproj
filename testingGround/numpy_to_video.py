import ffmpeg
import numpy as np

def vidwrite(fn, frames, framerate=60, vcodec='libx264'):
    if not isinstance(frames, np.ndarray):
        frames = np.asarray(frames)
    n,height,width,channels = frames.shape
    process = (
        ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
            .output(fn, pix_fmt='yuv420p', vcodec=vcodec, r=framerate)
            .overwrite_output()
            .run_async(pipe_stdin=True)
    )
    for frame in frames:
        process.stdin.write(
            frame
                .astype(np.uint8)
                .tobytes()
        )
    process.stdin.close()
    process.wait()

vidwrite("testOutputs/concated.mp4",  np.load('inputs/concated.npy'))