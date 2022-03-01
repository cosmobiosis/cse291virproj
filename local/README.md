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

## Execute For Memory Profiling
    conda activate deepeye_env
    cd execute_v1_improved
    mprof run mprof_execute.py
    mprof plot
    python normal_execute.py

## Output
![alt text](https://raw.githubusercontent.com/cosmobiosis/vir291proj/main/local/execute_v1_improved/memory_obj.jpg)
![alt text](https://raw.githubusercontent.com/cosmobiosis/vir291proj/main/local/execute_v1_improved/stages.png)
