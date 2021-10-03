# encoding: utf-8

import functools

import torch
import torch.nn as nn


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator_mnist(nn.Module):
    def __init__(self, input_c=1, output_nc=1, num_downs=7, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, output_function=nn.Tanh):
        super(UnetGenerator_mnist, self).__init__()
        # construct unet structure
        self.unet_block_real_img = UnetPre(input_c=1)
        self.unet_block_sec_img = UnetPre(input_c=1)

        self.model = nn.Sequential(
            nn.Conv2d(6,ngf, kernel_size=3, stride=1, padding=1,bias=True),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(),
            nn.Conv2d(ngf,ngf, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(),
            nn.Conv2d(ngf,ngf, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(),
            nn.Conv2d(ngf, input_c, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Tanh()
        )


    def forward(self, input, sec_img):
        real_img_feature = self.unet_block_real_img(input)
        sec_img_feature = self.unet_block_sec_img(sec_img)
        contain_img = torch.cat([real_img_feature, sec_img_feature], dim=1)
        output = self.model(contain_img)
        return output
class UnetGenerator(nn.Module):
    def __init__(self, input_nc=6, output_nc=3, num_downs=6, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, output_function=nn.Tanh):
        super(UnetGenerator, self).__init__()
        # construct unet structure
        self.unet_block_real_img = UnetPre(input_c=3)
        self.unet_block_sec_img = UnetPre(input_c=3)
        
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        #for i in range(num_downs - 5):
            #unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 #norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer, output_function=output_function)

        self.model = unet_block
    def forward(self, input, sec_img):

        real_img_feature = self.unet_block_real_img(input)
        sec_img_feature = self.unet_block_sec_img(sec_img)
        contain_img = torch.cat([real_img_feature, sec_img_feature], dim=1)
        output = self.model(contain_img)
        return output



class UnetPre(nn.Module):
    def __init__(self, input_c):
        super(UnetPre, self).__init__()
        '''
        self.pre_prepare1 = nn.Sequential(
            nn.Linear(1600, 1400),
            nn.BatchNorm1d(1400),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(1400, 1200),
            nn.BatchNorm1d(1200),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(1200, 1 * 32 * 32),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        '''

        self.pre = nn.Sequential(
            nn.ConvTranspose2d(input_c, 3, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ConvTranspose2d(3, 3, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))


    def forward(self, input):
        out = self.pre(input)
        return out


class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, output_function=nn.Sigmoid):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            if output_function == nn.Tanh:
                up = [uprelu, upconv, nn.Tanh()]
            else:
                up = [uprelu, upconv, nn.Sigmoid()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)
