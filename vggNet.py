from torch import nn

class VGG(nn.Module):
    def __init__(self, model: str = 'A', add_norm: bool = False, drop_p: float = 0.5, n_classes: int = 1000):
        super().__init__()
        
        self.convnet = self.createLayers(model, add_norm)
        self.maxpool = nn.AdaptiveMaxPool2d(output_size = (7, 7))
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096),
                                        nn.Dropout(drop_p),
                                        nn.Linear(4096, 4096),
                                        nn.Dropout(drop_p),
                                        nn.Linear(4096, n_classes))
    
    def forward(self, x):
        x = self.convnet(x)
        # print(x.shape)
        x = self.maxpool(x)
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)
        x = self.classifier(x)
        
        return x
    
    def createLayers(self, model: str = 'A', add_norm: bool = False):
        model_configs = self.getConfigs(model)
        
        layers = []
        in_filters = 3
        for out_filter in model_configs:
            if type(out_filter) == int:
                layers += self.convBlock(in_features = in_filters, out_features = out_filter, add_norm = add_norm)
                in_filters = out_filter
                
            elif out_filter == 'm':
                layers += [nn.MaxPool2d(kernel_size = 2, stride = 2)]
                
            elif 'c' in out_filter:
                out_filter = int(out_filter.split('-')[1])
                layers += self.convBlock(in_features = in_filters, out_features = out_filter, 
                                         kernel_size = 1, padding = 0, add_norm = add_norm)
        
        return nn.Sequential(*layers)
    
    def convBlock(self, in_features, out_features, kernel_size: str = 3, padding: int = 1, add_norm: bool = False):
        if add_norm:
            block = [nn.Conv2d(in_features, out_features, kernel_size = kernel_size, stride = 1, padding = padding),
                     nn.BatchNorm2d(out_features),
                     nn.ReLU()]
        else:
            block = [nn.Conv2d(in_features, out_features, kernel_size = kernel_size, stride = 1, padding = padding),
                     nn.ReLU()]
        
        return block
    
    def getConfigs(self, model: str = 'A'):
        configs = {
            'A': [64, 'm', 128, 'm', 256, 256, 'm', 512, 512, 'm', 512, 512],
            'B': [64, 64, 'm', 128, 128, 'm', 256, 256, 'm', 512, 512, 'm', 512, 512, 'm'],
            'C': [64, 64, 'm', 128, 128, 'm', 256, 256, 'c-256', 'm', 512, 512, 'c-512', 'm', 512, 512, 'c-512'],
            'D': [64, 64, 'm', 128, 128, 'm', 256, 256, 256, 'm', 512, 512, 512, 'm', 512, 512, 512],
            'E': [64, 64, 'm', 128, 128, 'm', 256, 256, 256, 256, 'm', 512, 512, 512, 512, 'm', 512, 512, 512, 512],
        }
        model_selection = configs[model]
        
        return model_selection