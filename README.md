# SwinUNETR Covid segmentation
## Overview
### About
This project aims to compare transformer-based and fully-convolutional segmentation models.
We are comparing five models (three fully-convolutional and two transformer-based):
  - U-Net
  - Basic U-Net
  - Flexible U-Net
  - UNETR
  - SwinUNETR

### Data
Models are trained on [COVID-19 lesion segmentation dataset](https://www.kaggle.com/datasets/maedemaftouni/covid19-ct-scan-lesion-segmentation-dataset). This dataset merges the COVID-19 lesion masks and their corresponding frames of these 3 public datasets, with 2729 image and ground truth mask pairs.

## Results
The results of the research are presented in the following table and graph:
| Model name     | Recall | Precision | F1 | IOU | GDL | Dice | Execution time (s.) |
|----------------|--------|-----------|----|-----|-----|------|---------------------|
| U-Net          |  0.64  |    0.63   |0.55| 0.46| 0.59| 0.42 |          180        |
| Basic U-Net    |  0.81  |    0.69   |0.72| 0.58| 0.72| 0.29 |          1980       |
| Flexible U-Net |  0.74  |    0.77   |0.72| 0.59| 0.72| 0.28 |          2580       |
| UNETR          |  0.79  |    0.76   |0.76| 0.64| 0.76| 0.24 |          3200       |
| SwinUNETR      |  0.81  |    0.78   |0.78| 0.66| 0.77| 0.23 |          3600       |

![Comparison graph](https://github.com/IlliaOl/Swin-UNETR_Covid_segmentation/assets/77388859/25a4102e-a53a-443b-bc28-824b1503a9f5)

