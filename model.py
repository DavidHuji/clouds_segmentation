from models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
import torch
from torch import nn
import macros

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def get_weights_initializer(std_hp=1.0,Xaviar_init=False):
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if Xaviar_init:
                torch.nn.init.xavier_normal_(m.weight)
            else:
                torch.nn.init.normal_(m.weight, mean=0.0, std=std_hp)
    return weights_init


def initialize_weights(net, Xaviar_init=False, std_hp=1.0):
    net.apply(get_weights_initializer(std_hp=std_hp))
    if Xaviar_init:
        net.apply(get_weights_initializer(std_hp=std_hp, Xaviar_init=True))


def getModel(outputchannels=1, using_unet=False, train_all=True):
    random_init = True
    if using_unet:
        random_init = (outputchannels != 1)
        model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                               in_channels=1 if macros.one_ch_in else 3, out_channels=outputchannels, init_features=32, pretrained=(not random_init))

    else:
        model = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True)

    if train_all == False:
        for param in model.parameters():
            param.requires_grad = False

    if not using_unet:
        model.classifier = DeepLabHead(2048, outputchannels)

    if macros.using_michals_unet:
        model = michals_unet(1 if macros.one_ch_in else 3, outputchannels)

    if random_init:
        initialize_weights(model, Xaviar_init=True)

    if macros.use_initialisation_weights:
        #  assuming weights are from the same model
        model.load_state_dict(torch.load(macros.initialisation_weights, map_location=torch.device(device)))
    model.train()
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def michals_unet(in_ch, out_ch):
    class UNET(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()

            self.conv1 = self.contract_block(in_channels, 32, 7, 3)
            self.conv2 = self.contract_block(32, 64, 3, 1)
            self.conv3 = self.contract_block(64, 128, 3, 1)

            self.upconv3 = self.expand_block(128, 64, 3, 1)
            self.upconv2 = self.expand_block(64 * 2, 32, 3, 1)
            self.upconv1 = self.expand_block(32 * 2, out_channels, 3, 1)

        def __call__(self, x):
            # downsampling part
            conv1 = self.conv1(x)
            conv2 = self.conv2(conv1)
            conv3 = self.conv3(conv2)

            upconv3 = self.upconv3(conv3)

            upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
            upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

            return upconv1

        def contract_block(self, in_channels, out_channels, kernel_size, padding):
            contract = nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(),
                torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )

            return contract

        def expand_block(self, in_channels, out_channels, kernel_size, padding):
            expand = nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
                                   torch.nn.BatchNorm2d(out_channels),
                                   torch.nn.ReLU(),
                                   torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
                                   torch.nn.BatchNorm2d(out_channels),
                                   torch.nn.ReLU(),
                                   torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2,
                                                            padding=1, output_padding=1)
                                   )
            return expand
    return UNET(in_ch, out_ch)


if __name__ == '__main__':
    print(getModel(outputchannels =3, using_unet=True, train_all=True))
    # print(count_parameters(createDeepLabv3(outputchannels =3, using_unet=True, train_all=True)))
    # print((createDeepLabv3(using_unet=True, train_all=True)))
    # print((michals_unet(3,4)))
