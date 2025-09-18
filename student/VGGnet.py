import torch
import torch.nn as nn
import torch.nn.functional as F

class EmoVGGVoxStudent(nn.Module):
    """
    VGG-M style student for 4s spectrograms, following Albanie et al.
    Input: (B, 1, 512, T). With T≈400, pool5 -> (B, 256, 9, 11).
    """
    def __init__(self, num_classes=8):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 96, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(96)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn2   = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4   = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5   = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(kernel_size=(5,3), stride=(3,2), padding=0)

        self.fc6   = nn.Conv2d(256, 4096, kernel_size=(9,1), stride=1, padding=0, bias=False)
        self.bn6   = nn.BatchNorm2d(4096)

        self.apool = nn.AdaptiveAvgPool2d((1,1))

        self.fc7   = nn.Conv2d(4096, 1024, kernel_size=1, bias=False)
        self.bn7   = nn.BatchNorm2d(1024)
        self.fc8   = nn.Conv2d(1024, num_classes, kernel_size=1, bias=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool5(x) 
    
        if x.shape[-2] != 9:
            raise RuntimeError(f"Expected freq height=9 before fc6, got {x.shape[-2]}")

        x = F.relu(self.bn6(self.fc6(x)))   
        x = self.apool(x)                   
        x = F.relu(self.bn7(self.fc7(x)))   
        x = self.fc8(x)                     
        return x.view(x.size(0), -1)        

if __name__ == "__main__":
    import torch
    m = EmoVGGVoxStudent(8)
    x = torch.randn(2, 1, 512, 400)
    y = m(x)
    print("OK, logits shape =", y.shape)