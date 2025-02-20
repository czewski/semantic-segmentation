## Experiment 3
### Method: 
Loading pre-trained weights from "deeplabv3_resnet50".
    - True, False

### Purpose: 
Using transfer learning generally improves model performance. 

### Experimental process: 
Same process from previous experiment, but with the following parameters: 

```
        --root 'data' \
        --batch_size 8 \
        --epoch 10 \
        --lr 0.001 \
        --mask_percentage 0.3 \
        --resize_to  256 \
        --gamma $gamma
```

### Results

#### Stats 

| Pre-Trained Weights | Loss    |  
| :---:   | :---: | 
| True  | 0.2040  |  
| False | 0.2718 | 

#### Conclusion

Importing weights from Resnet50 model improves the performance by ~33% (in terms of loss value).

Checking the PyTorch page, they say that "These weights were trained on a subset of COCO, using only the 20 categories that are present in the Pascal VOC dataset". 