## Implementation Details

I usually try to follow this same structure of files, as it worked really well when doing my masters project. (Having separated folders for custom loss, custom metrics, utils).

In the main runner (main.py) I concatenated all steps, trying to keep it as clean as posible, but still let comments (I like to keep comments for the shape of tensors inside training loop, and inside the model forward function).

The LoveDA dataset didn't had masks for test data, so I opted to not use it right now, a solution that I could've used was to re-split the validation set, but I opted not to do it, as the validation and training loss can give at least some clues of how the performance of the training is doing. 

The correct approach would be to implement a metric like IoU (intersection over union) for performance analysis. 

### Random Points in Mask

First I tried a very slow approach, that used 2 nested loops. 
```
    for item_batch in range(mask.shape[0]):
      for point in range(num_points):
```
In this approach I just populate a new mask with generated random coordinates. 

Then I searched for a more efficient approach, and got: 
```
    rand_mask = torch.rand(B, H, W, device=mask.device)  # (B, H, W)
    topk_indices = torch.topk(rand_mask.flatten(1), num_points, dim=1).indices  # (B, num_points)
    binary_mask = torch.zeros(B, H * W, device=mask.device)  # (B, H * W)
    binary_mask.scatter_(1, topk_indices, 1.0)  # (B, H * W)
    binary_mask = binary_mask.view(B, C, H, W)  # (B, C, H, W)
    result = (mask - 1) * binary_mask + 255 * (1 - binary_mask)
```
This approach generates a random bitwise mask, then applies it to the origin mask. 

##### How? 
First, generate random values with torch.rand(). 
Then the topk function is used to filter out the higher values from the random generation (using num_points to respect the max values).
A placeholder zeros mask is created.
The scatter() function is used to apply all active values from the topk indices as 1 in the binary mask.
The view() operation readjust the shape to respect the original mask shape. 
And finally we can copy the values from the original mask, to the positive elements in the binary mask, (if the value is 1, copy the mask value-1, if the value is 0, the value goes to 255).

I had some weird problems using -1 or 0, so I decided to use 255 as the ignore_index, (the pCE loss implementation uses the 255 value to create the GT labeled mask, and to ignore the index in the F.CE)

I'm reducing 1 from the mask value (classes), so the model also ignores the class 0 (background). 


## TODO: Comparison between some prediction and ground truth: 
