# meriti-technical-assessment

## TODO: 
- Loss isnt enough, need metrics (experiment 1 rerun)

- Run other experiments
- Histogram matching
- Data augmentation? 
- Check if mask is returning zeros or 255
- resplit valid to test (because test dont have mask)
- Data analysis (description of classes, dimensions, number of images, split)
- Understand better the random point process
- Image visualization (ground truth mask x predicted mask) - https://www.kaggle.com/code/mahastudentbazouzi/track4
- Organize tech reports 

## Task 1
- [Implementation: Partial Cross Entropy](loss/partial_cross_entropy.py)
- [Theory](reports/pce_analysis.md)

## Task 2
- [Runner](main.py)
- [Implementation Details](reports/imp_details.md)

## Task 3 
- [Data Analysis](reports/data_analysis.md)

- [Experiments](run.sh)
    - Change percentage of points [0.1, 0.3, 0.7, 0.1]
        - [Report Experiment 1](reports/report_experiment1.md)
    - Image preprocessing? [256, 128, 64], rescaling?
        - [Report Experiment 2](reports/report_experiment2.md)
    - Data augmentation? 
        - [Report Experiment 3](reports/report_experiment3.md)
    - Loading pre-trained weights 
        - [Report Experiment 4](reports/report_experiment4.md)


Do at least one inference, and check visuals

- Results: 
    - [Stats](stats/data.csv)
    - [Losses](loss_curves/loss_curve.png)
    - [Checkpoints](checkpoints/test.pth)

Talk about: 
- training being unstable with low points in mask

- Data description
    - I used only the urban data (maybe add rural?)


## Extra: 
- Add metrics?
- Unbalanced data? Maybe use wcce (weighted categorical cross-entropy)
- Use test data as visualization/inference (dont have masks)

## References 
- https://hal.science/hal-04330824
- https://arxiv.org/pdf/2312.05391v1 - Categorical losses
- https://arxiv.org/pdf/1708.02002 - Focal loss
- https://arxiv.org/pdf/2306.16252v1 - Land cover segmentation (sparse annotation)]
- https://www.researchgate.net/publication/355390292_LoveDA_A_Remote_Sensing_Land-Cover_Dataset_for_Domain_Adaptive_Semantic_Segmentation - LoveDA dataset
- https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w4/Kuo_Deep_Aggregation_Net_CVPR_2018_paper.pdf - Land Cover segmentation classification
- https://forum.image.sc/t/partial-annotation-for-deep-learning-semantic-segmentation/90229 - Partial annotation 