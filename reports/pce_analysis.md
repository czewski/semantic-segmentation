## Implementation: 

I followed the steps described in this paper: https://arxiv.org/pdf/1708.02002, for the Focal Loss, and combined with the formula in the assessment pdf. 

```
pCE = sum (focal loss (pre, gt) * mask) / sum (mask) + 0.00001 
focal loss = - alpha * mod_factor * log(pt)  
mod_factor = (1 - pt)^gamma
- log(pt) = CEloss(p,y)
```

![Focal Loss](https://i.imgur.com/2n7ZsN0.png)

