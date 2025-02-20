# Simple analysis of main dataset attributes 

## Description of data
The [LoveDA Dataset](https://www.researchgate.net/publication/355390292_LoveDA_A_Remote_Sensing_Land-Cover_Dataset_for_Domain_Adaptive_Semantic_Segmentation) contain urban and rural remote sensing images from cities of China. 

>  Basic statistics: 

| Attempt | Number of Images    | Number of Masks    |
| :---:   | :---: | :---: | 
| Train | 1156   | 1156   | 
| Valid | 677   | 677   |
| Test | 820   | -   |
| Total | 2653   | 1833   |

> Example of Data (image / mask)
![Example of data](https://i.imgur.com/f0cxYxh.png) 
![Example of data 1](https://i.imgur.com/eKLJgta.png) 

The image is a RGB file, with dimension (1024 x 1024), the mask have the same dimension but is in grayscale, where each intensity corresponds to a class.

## Classes: 

| Class | Value    | 
| :---:   | :---: | 
| background | 1   | 
| building | 2   | 
| road | 3   | 
| water | 4   | 
| barren | 5   | 
| forest | 6   | 
| agriculture | 7   | 

## Distribution of Classes (annotations)
In the following pictures we can observe a dominance of the "background" class in both sets. In the validation set we see a lot of annotations in the "agriculture" class. 

![Train set class distribution](https://i.imgur.com/vsSzdbW.png) 

![Valid set class distribution](https://i.imgur.com/5rTVpCG.png) 


