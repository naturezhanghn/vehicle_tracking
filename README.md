# 介绍
Python接口的交通检测跟踪工具
## 用法
* 使用镜像
    ```
    reg.supremind.info/algorithmteam/supreimage/tools/pythontraffictracking:pytorch1.4-trt6.0.1-ubuntu18.04-cuda10.2

    ```
* atom启容器
    ```
    直接选择镜像址址中的镜像启工作台
    ```
* 裸机启容器（可选）
    ```
    sudo docker login reg.supremind.info 
    #输入gitlab username and password
    sudo docker pull reg.supremind.info/algorithmteam/supreimage/tools/pythontraffictracking:pytorch1.4-trt6.0.1-ubuntu18.04-cuda10.2
    sudo nvidia-docker run -d --shm-size 512G -v /data:/workspace -i 'image_id'  /bin/bash
    sudo docker exec -it 'container_id' bash
    ```
* 代码下载
    ```
    cd /workspace
    git clone https://git.supremind.info/algorithmteam/supreimage/tools/pythontraffictracking.git
    cd pythontraffictracking
    ```
* 模型下载
    ```
    smcp sm/supredata-internal-algorithm/models/TRAFFIC_DET/norelease/PeleeNet_ATSSv2_4class_det_20210311_e29cd266_OPENCV_960cut8/PeleeNet_ATSSv2_4class_det_20210311_e29cd266_OPENCV_960cut8-v2.13.4.tronmodel models
    ```
* 视频下载
    ```
    smcp sm/supredata-internal-algorithm/TEST/example.mp4 videos
    ```
* 运行
    ```
    python tron_tracker.py
    ```
    若要生成[评测工具](https://git.supremind.info/algorithmteam/supreimage/tools/track_analysis_tools) 依赖的输入格式，则运行：
    ```
    python tron_tracker.py --save_txt
    ```
## todo
- [X] 基础交通检测跟踪推理功能
- [] 增加交通属性推理功能
- [] 增加交通车牌推理功能
- [] 支持Suprevison推理功能# vehicle_tracking
