## Experiment 3
### Method: 
Loading pre-trained weights from "deeplabv3_resnet50".
    - True, False

### Purpose: 
Using transfer learning generally improves model performance. 

### Experimental process: 
Using a bash script to change the values by argument parsing in the python main runner.

### Results

#### Stats 

| Pre-Trained Weights | Loss    | IoU    | 
| :---:   | :---: | :---: | 
| True  | 0.2040  | -   |  
| False | 0.2718 | -   |

#### Conclusion

Importing weights from Resnet50 model improves the performance by ~33% (in terms of loss value).

Checking the PyTorch page, they say that "These weights were trained on a subset of COCO, using only the 20 categories that are present in the Pascal VOC dataset". 