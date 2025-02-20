## Experiment 2
### Method: 
Using different image sizes during training.
    - 64 128 256 512

### Purpose: 
I think that the image size won't interfere much in the training results, but smaller images may make the model train faster. 

### Experimental process: 
Same process from previous experiment, but with the following parameters: 

```
        --root 'data' \
        --batch_size 8 \
        --epoch 10 \
        --lr 0.001 \
        --mask_percentage 0.3 \
        --resize_to  $resize \
        --gamma 2
```

### Results

#### Loss curve

Similar to Experiment 1, here we can also see a bit of disturbance during training. Following the same logic, validation and train losses are decreasing, with a "small" gap between curves. 

    - 64?

![64](https://i.imgur.com/VZumd8L.png)

    - 128 

![128](https://i.imgur.com/H6HDbyg.png)

    - 256

![256](https://i.imgur.com/X63ikp2.png)
    
    - 512 (This took around 10 hours to run for 5 epochs)

![512](https://i.imgur.com/aIwpQ3a.png)

#### Stats 

| Image Size | Loss    | Time (min) | 
| :---:   | :---: | :---: | 
| 64  | -   |  -   | 5.8  | 
| 128  | 0.2341   |  12.9   | 
| 256 | 0.2089  |  19.7  | 
| 512 | 0.3127 |  580.8  | 

#### Conclusion

Having bigger resolution dramatically improve the performance of the model at computational cost. Even that I was only able to train just for 5 epochs with 512x512, we can observe in the loss curve that the training was very linear, without the spikes shown in other figures, and even with the high loss value the model have room for improvements with more training time. 





