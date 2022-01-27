import ffmpeg

in_file0 = ffmpeg.input('inputs/countdown.mp4')
in_file1 = ffmpeg.input('inputs/countdown_frame.mp4')
(
    ffmpeg
    .concat(
        in_file0.trim(start_frame=10, end_frame=20),
        in_file1.trim(start_frame=0, end_frame=10),
    )
    .output('testOutputs/out.mp4')
    .run()
)
print("Finished")