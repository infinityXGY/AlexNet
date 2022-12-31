import torch
import csv
from PIL import Image
from torch import nn
from torchvision.transforms import transforms

classes = ["female" , "male" ]  #这里改成自己的种类即可，从左到右对应自己的训练集种类排序的从左到右

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))
class alexnet(nn.Module):
    def __init__(self):
        super(alexnet, self).__init__()
        self.model = nn.Sequential(

            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 120, 120]  output[48, 55, 55]

            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),  # output[128, 27, 27]

            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),  # output[192, 13, 13]

            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),  # output[192, 13, 13]

            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),  # output[128, 13, 13]

            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 6, 6]
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 7),

        )

    def forward(self, x):
        x = self.model(x)
        return x

# alexnet1 = alexnet()
#
# alexnet1.load_state_dict(torch.load("alexnet.pth", map_location=device))#训练得到的alexnet模型放入当前文件夹下
#
# outputs = alexnet1(image)
#
# ans = (outputs.argmax(male)).item()
# print(classes[ans])

p = 3000
for i in range(1083):
    image_path = "test/testimages/"+str(p)+".jpg"  # 需要测试的图片放入当前文件夹下，这里改成自己的图片名即可
    trans = transforms.Compose([transforms.Resize((120, 120)),
                                transforms.ToTensor()])
    image = Image.open(image_path)

    image = image.convert("RGB")
    image = trans(image)

    image = torch.unsqueeze(image, dim=0)

    alexnet1 = alexnet()

    alexnet1.load_state_dict(torch.load("alexnet.pth", map_location=device))  # 训练得到的alexnet模型放入当前文件夹下

    outputs = alexnet1(image)

    ans = (outputs.argmax(1)).item()
    print(image_path)
    print(classes[ans])
    p=p+1