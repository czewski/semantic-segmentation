## Experiment 2
### Method: 
Using different image sizes during training.
    - 64 128 256 512

### Purpose: 
I think that the image size won't interfere much in the training results, but smaller images may make the model train faster. 

### Experimental process: 
Using a bash script to change the values by argument parsing in the python main runner.

### Results

#### Loss curve

Similar to Experiment 1, here we can also see a bit of disturbance during training. Following the same logic, validation and train losses are decreasing, with a "small" gap between curves. 

    - 64?

![10]()

    - 128 

![10]()

    - 256

![30]()
    
    - 512 (This took around 5 hours to run for 5 epochs, and then my computer ran out of VRAM)

![70]()

#### Stats 

| Image Size | Loss    | IoU    | 
| :---:   | :---: | :---: | 
| 64  | -   | -   |  
| 128  | 0.2341   | -   |  
| 256 | 0.2089  | -   |
| 512 |   - | -   | 

#### Conclusion

Having bigger resolution dramatically improve the performance of the model. 





