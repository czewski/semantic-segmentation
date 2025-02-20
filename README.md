# Meriti Technical Assessment

## TODO: 

## Task 1
- [Partial Cross Entropy Code](loss/partial_cross_entropy.py)
- [How I implemented](reports/pce_analysis.md)

## Task 2
- [Runner](main.py)
- [Implementation Details](reports/imp_details.md)

## Task 3 
- [Data Analysis](reports/data_analysis.md)

- [Experiments](run.sh)
    - Change percentage of points [0.1, 0.3, 0.7, 0.1]
        - [Report Experiment 1](reports/report_experiment1.md)
    - Image preprocessing? [512, 256, 128, 64]
        - [Report Experiment 2](reports/report_experiment2.md)
    - Loading pre-trained weights [True, False]
        - [Report Experiment 3](reports/report_experiment3.md)
    - Changing gamma inside pCE [1, 2, 3]
        - [Report Experiment 4](reports/report_experiment4.md)
    <!-- - Data augmentation? 
        - [Report Experiment 5](reports/report_experiment5.md) -->

- Results: 
    - [Inference](inference.py)
    ![Inference](https://i.imgur.com/YrcBwVp.png)
    - [Stats](stats/)
    - [Losses](loss_curves/)
    - [Checkpoints](checkpoints/)

## Hypothesis to test in the future: 
- Change the value of "alpha" inside the pCE loss function. 
- Try using a "kernel" to simulate bigger point annotations. 
- Unbalanced data? Maybe use wcce (weighted categorical cross-entropy), having an alpha weight for each class. 
- Some data augmentation process (Histogram matching?). 
- Different model architectures. 

## References 
- https://arxiv.org/pdf/2312.05391v1 - Categorical losses
- https://arxiv.org/pdf/1708.02002 - Focal loss
- https://arxiv.org/pdf/2306.16252v1 - Land cover segmentation (sparse annotation)
- https://www.researchgate.net/publication/355390292_LoveDA_A_Remote_Sensing_Land-Cover_Dataset_for_Domain_Adaptive_Semantic_Segmentation - LoveDA dataset
- https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w4/Kuo_Deep_Aggregation_Net_CVPR_2018_paper.pdf - Land Cover segmentation classification
- https://forum.image.sc/t/partial-annotation-for-deep-learning-semantic-segmentation/90229 - Partial annotation 