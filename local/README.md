## Dependencies For Local Execution
    Tensorflow 1.13
    OpenCV
    Mprof
    Pympler

## Installing Steps
    conda create --name deepeye_env
    conda activate deepeye_env
    conda install tensorflow=1.13.1
    conda install opencv
    conda install memory_profiler
    conda install pympler

## Execute For Memory Profiling (One Run)
    conda activate deepeye_env
    cd execute_v1_improved
    mprof run mprof_execute.py
    mprof plot
    python normal_execute.py

## Execute For Memory Profiling (Multi-stage Garbage Collection)
    conda activate deepeye_env
    cd execute_v2_final_pipeline
    python mprof_pipeline_execute.py
    mprof plot

## Output
![alt text](https://raw.githubusercontent.com/cosmobiosis/deepeye_291_virtualization_proj/main/local/execute_v1_improved/memory_one_run.png)
![alt text](https://raw.githubusercontent.com/cosmobiosis/deepeye_291_virtualization_proj/main/local/execute_v2_final_pipeline/final_memory_stages.png)
![alt text](https://raw.githubusercontent.com/cosmobiosis/vir291proj/main/local/execute_v1_improved/memory_obj.jpg)
