import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F


model = smp.Unet(
    encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=2,  # model output channels (number of classes in your dataset)
)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.conv2 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=6,
            dilation=6,
            padding_mode="zeros",
        )
        self.conv3 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=12,
            dilation=12,
            padding_mode="zeros",
        )
        self.conv4 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=18,
            dilation=18,
            padding_mode="zeros",
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.out_conv = nn.Conv2d(
            out_channels * 5, out_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        size = x.shape[-2:]
        conv1 = F.relu(self.conv1(x))
        conv2 = F.relu(self.conv2(x))
        conv3 = F.relu(self.conv3(x))
        conv4 = F.relu(self.conv4(x))
        pool = F.relu(self.pool_conv(self.global_avg_pool(x)))
        pool = F.interpolate(pool, size=size, mode="bilinear", align_corners=False)
        return F.relu(
            self.out_conv(torch.cat([conv1, conv2, conv3, conv4, pool], dim=1))
        )


class FraSegNetVGG19(nn.Module):
    def __init__(self, num_classes=1):
        super(FraSegNetVGG19, self).__init__()
        # VGG19 Encoder
        self.encoder_block1 = self._conv_block(3, 64, 2)  # [64, 64]
        self.pool1 = nn.MaxPool2d(2, 2)  # 224 -> 112
        self.encoder_block2 = self._conv_block(64, 128, 2)  # [128, 128]
        self.pool2 = nn.MaxPool2d(2, 2)  # 112 -> 56
        self.encoder_block3 = self._conv_block(128, 256, 4)  # [256, 256, 256, 256]
        self.pool3 = nn.MaxPool2d(2, 2)  # 56 -> 28
        self.encoder_block4 = self._conv_block(256, 512, 4)  # [512, 512, 512, 512]
        self.pool4 = nn.MaxPool2d(2, 2)  # 28 -> 14
        self.encoder_block5 = self._conv_block(512, 512, 4)  # [512, 512, 512, 512]
        self.pool5 = nn.MaxPool2d(2, 2)  # 14 -> 7
        # ASPP
        self.aspp = ASPP(512, 256)  # 16x16
        # Decoder
        self.up4 = nn.ConvTranspose2d(256, 512, kernel_size=2, stride=2)  # 16 -> 32
        self.decoder_block4 = self._conv_block(512 + 512, 512, 2)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # 32 -> 64
        self.decoder_block3 = self._conv_block(256 + 256, 256, 2)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # 64 -> 128
        self.decoder_block2 = self._conv_block(128 + 128, 128, 2)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # 128 -> 224
        self.decoder_block1 = self._conv_block(64 + 64, 64, 2)
        # Output
        self.output_layer = nn.Conv2d(64, num_classes, kernel_size=1, stride=1)

    def _conv_block(self, in_channels, out_channels, num_layers):
        layers = []
        for _ in range(num_layers):
            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            )
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        x1 = self.encoder_block1(x)  # 224x224
        p1 = self.pool1(x1)  # 112x112
        x2 = self.encoder_block2(p1)  # 112x112
        p2 = self.pool2(x2)  # 56x56
        x3 = self.encoder_block3(p2)  # 56x56
        p3 = self.pool3(x3)  # 28x28
        x4 = self.encoder_block4(p3)  # 28x28
        p4 = self.pool4(x4)  # 14x14
        x5 = self.encoder_block5(p4)  # 14x14
        p5 = self.pool5(x5)  # 7x7
        # ASPP
        aspp_out = self.aspp(p5)  # 16x16
        # Decoder
        d4 = self.up4(aspp_out)  # 16 -> 32
        d4 = torch.cat([d4, x4], dim=1)
        d4 = self.decoder_block4(d4)
        d3 = self.up3(d4)  # 32 -> 64
        d3 = torch.cat([d3, x3], dim=1)
        d3 = self.decoder_block3(d3)
        d2 = self.up2(d3)  # 64 -> 128
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.decoder_block2(d2)
        d1 = self.up1(d2)  # 128 -> 224
        d1 = torch.cat([d1, x1], dim=1)
        d1 = self.decoder_block1(d1)
        # Output
        out = self.output_layer(d1)  # 224x224
        return out


# Instantiate and check sizes
model = FraSegNetVGG19()
x = torch.randn(1, 3, 224, 224)  # Batch size 1, RGB image
output = model(x)
print("Output shape:", output.shape)
