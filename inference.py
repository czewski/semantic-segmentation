import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.models.segmentation as models
from torchvision import transforms

model = models.deeplabv3_resnet50(pretrained=True)   
model.classifier[4] = torch.nn.Conv2d(256, 7, kernel_size=(1, 1), stride=(1, 1)) 
model.load_state_dict(torch.load("checkpoints/19_02_2025_14:59:52_test.pth"))
model.eval() 

compose = transforms.Compose([
    transforms.Resize((256, 256)),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

image_path = "data/test/images_png/5167.png"
image = Image.open(image_path).convert("RGB")  
input_tensor = compose(image).unsqueeze(0)  

with torch.no_grad():
    output = model(input_tensor)["out"]

pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()  

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Input Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(pred_mask)
plt.title("Predicted Mask")
plt.axis("off")
plt.show()