# meriti-technical-assessment

## TODO: 
- Make create mask more efficient
- Prepare for more data
- Plot training
- Check why training is unstable (probably need more data)
- I used only the urban data (maybe add rural?)
- And the no-data regions were assigned 0 which should be ignored., ok there was already a 0 class, maybe need to change this index as well

## Task 1
- Done pCE

## Task 2
- Done model
- Add correct data (split)
- Data Augmentation
- Run model

## Task 3 (Technical reports = method + purpose + process + results)
- Change percentage of points [0.1, 0.3, 0.7, 0.1]
- Image preprocessing? [256, 128, 64], rescaling?
- Data augmentation? 
- Histogram matching

## Extra: 
- Unbalanced data? Maybe use wcce (weighted categorical cross-entropy)
- Data analysis 
- Use test data as visualization/inference (dont have masks)
- Image visualization (ground truth mask x predicted mask) - https://www.kaggle.com/code/mahastudentbazouzi/track4


## References 
- https://hal.science/hal-04330824
- https://arxiv.org/pdf/2312.05391v1 - Categorical losses
- https://arxiv.org/pdf/1708.02002 - Focal loss
- https://arxiv.org/pdf/2306.16252v1 - Land cover segmentation (sparse annotation)]
- https://www.researchgate.net/publication/355390292_LoveDA_A_Remote_Sensing_Land-Cover_Dataset_for_Domain_Adaptive_Semantic_Segmentation - LoveDA dataset
- https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w4/Kuo_Deep_Aggregation_Net_CVPR_2018_paper.pdf - Land Cover segmentation classification
- https://forum.image.sc/t/partial-annotation-for-deep-learning-semantic-segmentation/90229 - Partial annotation 