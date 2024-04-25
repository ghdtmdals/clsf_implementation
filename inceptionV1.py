import torch
from torch import nn

class convBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.convBlock = nn.Sequential(nn.Conv2d(in_features, out_features, kernel_size = kernel_size, stride = stride, padding = padding),
                                       nn.BatchNorm2d(out_features),
                                       nn.ReLU())
    
    def forward(self, x):
        x = self.convBlock(x)
        
        return x

class naiveInception(nn.Module):
    def __init__(self, in_features: int, cv1_out: int, cv3_out: int, cv5_out: int, pool_out: int):
        super().__init__()
        self.branch1 = convBlock(in_features, cv1_out, kernel_size = 1, stride = 1)
        self.branch2 = convBlock(in_features, cv3_out, kernel_size = 3, stride = 1, padding = 1)
        self.branch3 = convBlock(in_features, cv5_out, kernel_size = 5, stride = 1, padding = 2)
        
        # Maxpool에 1x1 Convolution 없지만 Channel 수 맞추기 위해 추가
        self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size = (3, 3), stride = 1, padding = 1),
                                     convBlock(in_features, pool_out, kernel_size = 1, stride = 1))
        
    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        outputs = torch.cat([x1, x2, x3, x4], 1) # Concat on Channel axis
        # print(x1.shape, x2.shape, x3.shape, x4.shape)
        
        return outputs

class reducInception(nn.Module):
    def __init__(self, in_features: int, cv1_out: int, cv3_red: int, cv3_out: int, cv5_red: int, cv5_out: int, pool_out: int):
        super().__init__()
        self.branch1 = convBlock(in_features, cv1_out, kernel_size = 1, stride = 1)
        self.branch2 = nn.Sequential(convBlock(in_features, cv3_red, kernel_size = 1, stride = 1),
                                     convBlock(cv3_red, cv3_out, kernel_size = 3, stride = 1, padding = 1))
        self.branch3 = nn.Sequential(convBlock(in_features, cv5_red, kernel_size = 1, stride = 1),
                                     convBlock(cv5_red, cv5_out, kernel_size = 5, stride = 1, padding = 2))
        self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size = (3, 3), stride = 1, padding = 1),
                                     convBlock(in_features, pool_out, kernel_size = 1, stride = 1))
        
    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        outputs = torch.cat([x1, x2, x3, x4], 1)
        
        return outputs
    
class InceptionAuxilary(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_classes: int = 1000):
        super().__init__()
        self.avgpool = nn.AvgPool2d(kernel_size = 5, stride = 3)
        self.conv = convBlock(in_features, out_features, kernel_size = 1, stride = 1)
        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Linear(4 * 4 * out_features, 1024),
                                        nn.ReLU(),
                                        nn.Dropout(0.7),
                                        nn.Linear(1024, n_classes))
        
    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv(x)
        x = self.classifier(x)
        
        return x

class InceptionV1(nn.Module):
    def __init__(self, n_classes: int = 1000, module_option: str = 'reduction'):
        super().__init__()
        # Define Layers
        # 112, 112, 64
        self.conv1 = convBlock(3, 64, kernel_size = 7, stride = 2, padding = 3)
        
        # 56, 56, 64
        self.maxpool1 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        # 56, 56, 192
        self.conv2_1 = convBlock(64, 64, kernel_size = 1, stride = 1)
        self.conv2_2 = convBlock(64, 192, kernel_size = 3, stride = 1, padding = 1)
        
        # 28, 28, 192
        self.maxpool2 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        # 28, 28, 256
        self.inception3a = self.inceptionBlock(192, cv1_out = 64, cv3_out = 128, cv5_out = 32, 
                                               cv3_red = 96, cv5_red = 16, pool_out = 32, module_option = module_option)
        
        # 28, 28, 480
        self.inception3b = self.inceptionBlock(256, cv1_out = 128, cv3_out = 192, cv5_out = 96,
                                               cv3_red = 128, cv5_red = 32, pool_out = 64, module_option = module_option)
        
        # 14, 14, 480
        self.maxpool3 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        # 14, 14, 512
        self.inception4a = self.inceptionBlock(480, cv1_out = 192, cv3_out = 208, cv5_out = 48,
                                               cv3_red = 96, cv5_red = 16, pool_out = 64, module_option = module_option)
        
        # 14, 14, 512
        self.inception4b = self.inceptionBlock(512, cv1_out = 160, cv3_out = 224, cv5_out = 64,
                                               cv3_red = 112, cv5_red = 24, pool_out = 64, module_option = module_option)
        
        # 14, 14, 512
        self.inception4c = self.inceptionBlock(512, cv1_out = 128, cv3_out = 256, cv5_out = 64,
                                               cv3_red = 128, cv5_red = 24, pool_out = 64, module_option = module_option)
        
        # 14, 14, 528
        self.inception4d = self.inceptionBlock(512, cv1_out = 112, cv3_out = 288, cv5_out = 64,
                                               cv3_red = 144, cv5_red = 32, pool_out = 64, module_option = module_option)
        
        # 14, 14, 832
        self.inception4e = self.inceptionBlock(528, cv1_out = 256, cv3_out = 320, cv5_out = 128,
                                               cv3_red = 160, cv5_red = 32, pool_out = 128, module_option = module_option)
        
        # 7, 7, 832
        self.maxpool4 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        # 7, 7, 832
        self.inception5a = self.inceptionBlock(832, cv1_out = 256, cv3_out = 320, cv5_out = 128,
                                               cv3_red = 160, cv5_red = 32, pool_out = 128, module_option = module_option)
        
        # 7, 7, 1024
        self.inception5b = self.inceptionBlock(832, cv1_out = 384, cv3_out = 384, cv5_out = 128,
                                               cv3_red = 192, cv5_red = 48, pool_out = 128, module_option = module_option)
        
        self.classifier = nn.Sequential(nn.AvgPool2d(kernel_size = 7, stride = 1), # 1, 1, 1024
                                        nn.Flatten(start_dim = 1),
                                        nn.Dropout(0.4),
                                        nn.Linear(1024, n_classes)) # 1, 1, 1000
        
        # Auxilary 14, 14, 512 -> 4, 4, 512
        self.aux1 = InceptionAuxilary(512, 128, n_classes = n_classes)
        
        # Auxilary 14, 14, 528 -> 4, 4, 528
        self.aux2 = InceptionAuxilary(528, 128, n_classes = n_classes)
    
    def forward(self, x):
        # Define Forward Pass
        # 112, 112, 64
        x = self.conv1(x)
        
        # 56, 56, 64
        x = self.maxpool1(x)
        
        # 56, 56, 192
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        
        # 28, 28, 192
        x = self.maxpool2(x)
        
        # 28, 28, 256
        x = self.inception3a(x)
        
        # 28, 28, 480
        x = self.inception3b(x)
        
        # 14, 14, 480
        x = self.maxpool3(x)
        
        # 14, 14, 512
        x = self.inception4a(x)
        
        # Auxilary
        aux1 = self.aux1(x)
                
        # 14, 14, 512
        x = self.inception4b(x)
        
        # 14, 14, 512
        x = self.inception4c(x)
        
        # 14, 14, 528
        x = self.inception4d(x)
        
        # Auxilary
        aux2 = self.aux2(x)
        
        # 14, 14, 832
        x = self.inception4e(x)
        
        # 7, 7, 832
        x = self.maxpool4(x)
        
        # 7, 7, 832
        x = self.inception5a(x)
        
        # 7, 7, 1024
        x = self.inception5b(x)
        
        # Linear Classifier
        x = self.classifier(x) # need to split inner blocks
        
        return x, aux1, aux2
    
    def inceptionBlock(self, in_features: int, cv1_out: int, cv3_out: int, cv5_out: int,
                       pool_out: int, cv3_red: int = 0, cv5_red: int = 0, 
                       module_option: str = 'reduction'):
        if module_option == 'reduction':
            module = reducInception(in_features, cv1_out = cv1_out, cv3_red = cv3_red, cv3_out = cv3_out, 
                                    cv5_red = cv5_red, cv5_out = cv5_out, pool_out = pool_out)
            
        elif module_option == 'naive':
            module = naiveInception(in_features, cv1_out = cv1_out, cv3_out = cv3_out, cv5_out = cv5_out, pool_out = pool_out)
            
        return module
