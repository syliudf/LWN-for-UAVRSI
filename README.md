# LWN for UAVRSI
Light-Weight Semantic Segmentation Network for UAV Remote Sensing Images



# Briefly

- This repo introduces a light-weight semantic segmentation network for UAV Remote Sensing Images

- The network only requires 9M parameters 
- The experiments on the ISPRS Vaihingen dataset, UAVid dataset and UDD6 dataset had verify the effectiveness of it. 

# Environment

## Runtime environment

- Ubuntu 16.04
- PyTorch 1.6.0
- CUDA10.1+
- Nvidia GTX2080Ti

## Models

- All the models involved in `models/`

- Under the condition of the image size is 512x512, the performances of our models on the Vaihingen dataset are as follows:

    | Model |  mF1  | mIoU  |  OA   | Params(M) |
    | :---: | :---: | :---: | :---: | :-------: |
    |  LWN  | 86.79 | 77.11 | 88.27 |     9     |
    | LWN-A | 87.62 | 78.38 | 88.85 |    15     |

    UAVid:
    | Model | mIoU  |  OA   | Params(M) |
    | :---: | :---: | :---: | :-------: |
    |  LWN  | 67.82 | 87.13 |     9     |
    | LWN-A | 69.02 | 87.66 |    15     |
    
    UDD:
    | Model       |  mF1  | mIoU  |  OA   | Params(M) | 
    | :---------- | :---: | :---: | :---: | :-------: |
    | LWN         | 86.19 | 76.78 | 88.75 |   9       |
    | LWN-A       | 86.79 | 77.19 | 88.93 |   15      | 


​    

# Training

- It is recommended to make a new dir named `data`  and save or link the dataset under it. 

- Images and labels are recommended  to crop to `512*512`

- Then prepare the `data` as follows:

- ```
  data/uavid
  |-- train
  |   |-- image
  |	|	|-- seq1_000000.png
  |	|	|-- ...
  |   |-- label
  |	|	|-- seq1_000000.png
  |	|	|-- ...
  |-- val
  |   |-- image
  |	|	|-- seq16_000000.png
  |	|	|-- ...
  |   |-- label
  |	|	|-- seq16_000000.png
  |	|	|-- ...
  ```

-  Then set the parameters for training phase, such as `dataset`, `model_type` or  `data_root` on `config.ini`.

- `python main.py`