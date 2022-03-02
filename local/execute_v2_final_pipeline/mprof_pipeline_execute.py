from subprocess import PIPE, run

pipeline = ["func1_event_entry", 
	"func2_A1_frame_gray_generation", 
	"func2_B1_frame_generation", 
	"func2_A2_probability_mask_generation",
	"func2_A3_blob_locations_generation",
	"func3_pupil_iris_segmentation"
]

for name in pipeline:
	command = ['mprof', 'run', name + ".py"]
	run(command)