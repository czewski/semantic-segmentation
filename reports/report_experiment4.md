## Experiment 4
### Method: 
Changing values for the "gamma" inside the pCE loss function.
    - True, False

### Purpose: 
While reading the paper the author said that higher values improve the modulation factor inside focal loss, and using gamma = 0 is the same that using pure CE. 

### Experimental process: 
Using a bash script to change the values by argument parsing in the python main runner.

### Results

#### Stats 

| Pre-Trained Weights | Loss    | IoU    | 
| :---:   | :---: | :---: | 
| True  | 0.  | -   |  
| False | 0. | -   |

#### Conclusion
