from config import *
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))

class _Pool(nn.Sequential):
    def __init__(self, num_features):
        super(_Pool, self).__init__()
        self.add_module('pool_norm', nn.BatchNorm3d(num_features))
        self.add_module('pool_relu', nn.ReLU(inplace=True))
        self.add_module('pool', nn.Conv3d(num_features, num_features, kernel_size=2, stride=2))

class _Upsample(nn.Sequential):
    def __init__(self, nin, nout, ksize, stride):
        super(_Upsample, self).__init__()
        self.add_module('up', nn.ConvTranspose3d(nin, nin//2, kernel_size=ksize, stride=stride, padding=1, groups=nin//2, bias=False))
        self.add_module('norm', nn.BatchNorm3d(nin//2))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(nin//2, nout, kernel_size=1, stride=1, bias=False))


class DenseResNet(nn.Module):
    r"""DenseResNet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=16, block_config=(6, 12, 24, 16),
                 num_init_features=32, bn_size=4, drop_rate=0, num_classes=9):

        super(DenseResNet, self).__init__()

        # First three convolutions
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(2, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('conv1', nn.Conv3d(num_init_features, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm1', nn.BatchNorm3d(num_init_features)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv3d(num_init_features, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
        ]))
        self.features_bn = nn.Sequential(OrderedDict([
            ('norm2', nn.BatchNorm3d(num_init_features)),
            ('relu2', nn.ReLU(inplace=True)),
        ]))
        self.pool1 = nn.Conv3d(num_init_features, num_init_features, kernel_size=2, stride=2, padding=0,
                                         bias=False)


        self.denseblock1 =  _DenseBlock(num_layers=4, num_input_features=32, bn_size=4, growth_rate=16, drop_rate=0.2)
        self.transblock1 = _Transition(num_input_features=96, num_output_features=48)
        self.upblock4 = _Upsample(48, 32, ksize=4, stride=2) 

        self.pool2 = _Pool(num_features=48)
        self.denseblock2 =  _DenseBlock(num_layers=4, num_input_features=48, bn_size=4, growth_rate=16, drop_rate=0.2)
        self.transblock2 = _Transition(num_input_features=112, num_output_features=56)
        self.upblock3 = _Upsample(56, 48, ksize=4, stride=2) 

        self.pool3 = _Pool(num_features=56)
        self.denseblock3 =  _DenseBlock(num_layers=4, num_input_features=56, bn_size=4, growth_rate=16, drop_rate=0.2)
        self.transblock3 = _Transition(num_input_features=120, num_output_features=60)
        self.upblock2 = _Upsample(60, 56, ksize=4, stride=2)        

        self.pool4 = _Pool(num_features=60)
        self.denseblock4 =  _DenseBlock(num_layers=4, num_input_features=60, bn_size=4, growth_rate=16, drop_rate=0.2)
        self.transblock4 = _Transition(num_input_features=124, num_output_features=62)
        self.upblock1 = _Upsample(62, 60, ksize=4, stride=2) 


        # ----------------------- Classifier -----------------------
        self.bn_class = nn.BatchNorm3d(32)
        self.conv_class = nn.Conv3d(32, num_classes, kernel_size=1, padding=0)
        # ----------------------------------------------------------

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        conv3 = self.features(x)
        out = self.features_bn(conv3)

        out = self.pool1(out)
        out = self.denseblock1(out)
        tb1 = self.transblock1(out)

        out = self.pool2(tb1)
        out = self.denseblock2(out)
        tb2 = self.transblock2(out)

        out = self.pool3(tb2)
        out = self.denseblock3(out)
        tb3 = self.transblock3(out)

        out = self.pool4(tb3)
        out = self.denseblock4(out)
        out = self.transblock4(out)

        out = self.upblock1(out)
        out = torch.add(out, tb3)

        out = self.upblock2(out)
        out = torch.add(out, tb2)

        out = self.upblock3(out)
        out = torch.add(out, tb1)

        out = self.upblock4(out)
        out = torch.add(out, conv3)

        # ----------------------- classifier -----------------------
        out = self.conv_class(F.relu(self.bn_class(out)))
        # ----------------------------------------------------------

        return out


