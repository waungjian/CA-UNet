# CA-UNet
## Model training and inference
### The code and training weights for CA-UNet are placed in the $.\code$
### CA-UNet does not use pre-trained weights of Swin Transformer on Image-1K. If you want to train this model, you can use the following command:：
```C
python train.py
```
### The best weights we trained are stored in the path $.\code\model_out\$. You can reproduce the model's effects using the following command:：
```C
python test.py
```
## CA-UNet paper template
### We have placed the Latex template for this paper in the directory $.\paper\$.