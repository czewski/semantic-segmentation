## Experiment 4
### Method: 
Changing values for the "gamma" inside the pCE loss function.
    - 1, 2, 3

### Purpose: 
While reading the paper the author said that higher values improve the modulation factor inside focal loss, and using gamma = 0 is the same that using pure CE. 

### Experimental process: 
Using a bash script to change the values by argument parsing in the python main runner.

### Results

#### Stats 

| Pre-Trained Weights | Loss    | IoU    | 
| :---:   | :---: | :---: | 
| 1  | 0.3060  | -   |  
| 2 | 0.2040 | -   |
| 3 | 0.2109 | -   |

#### Conclusion

The obtained results follow the authors logic, better performance is obtained with "gamma = 2", and values closer to 0, basically ignores the modular part of the focal loss, resulting in higher loss values. 