import torch
import torch.nn as nn
from torch import Tensor
from typing import Type, Any, List, Optional, Union

"""
This file contains a complete, from-scratch implementation of the ResNet
architecture in PyTorch. It includes:
1.  BasicBlock: Used for ResNet-18 and ResNet-34.
2.  Bottleneck: Used for ResNet-50, ResNet-101, and ResNet-152.
3.  ResNet: The main class that assembles the blocks.
4.  Factory Functions: e.g., resnet50(), resnet34(), etc., to easily create
    specific model versions.
"""

class BasicBlock(nn.Module):
    """
    The Basic Residual Block used in ResNet-18 and ResNet-34.
    It consists of two 3x3 convolutional layers.
    """
    expansion: int = 1  # The expansion factor for the output channels

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride to use in the first 3x3 conv.
                          Used for downsampling.
            downsample (nn.Module, optional): A module to handle the
                          shortcut connection if dimensions change.
        """
        super(BasicBlock, self).__init__()
        
        # First convolutional layer
        # Bias is False because we use BatchNorm
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        # Save the input for the shortcut connection
        identity = x

        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Shortcut path (downsample if necessary)
        if self.downsample is not None:
            identity = self.downsample(x)

        # Add the shortcut to the main path output
        out += identity
        # Apply ReLU *after* the addition
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    The Bottleneck Residual Block used in ResNet-50, 101, and 152.
    It consists of three conv layers: 1x1 (reduce), 3x3, 1x1 (expand).
    """
    expansion: int = 4  # The expansion factor is 4 for Bottleneck blocks

    def __init__(
        self,
        in_channels: int,
        out_channels: int, # Note: this is the *bottleneck* channel count (e.g., 64)
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of *intermediate* channels (bottleneck).
                                The final output channels will be out_channels * 4.
            stride (int): Stride to use in the 3x3 conv.
            downsample (nn.Module, optional): Module for the shortcut if dimensions change.
        """
        super(Bottleneck, self).__init__()
        
        final_out_channels = out_channels * self.expansion

        # 1x1 Conv (Channel Reduction)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 3x3 Conv (Spatial Convolution)
        # This layer carries the stride for downsampling
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1 Conv (Channel Expansion)
        self.conv3 = nn.Conv2d(
            out_channels, final_out_channels, kernel_size=1, stride=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(final_out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        # Save the input for the shortcut connection
        identity = x

        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # Shortcut path (downsample if necessary)
        if self.downsample is not None:
            identity = self.downsample(x)

        # Add the shortcut to the main path output
        out += identity
        # Apply ReLU *after* the addition
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    The main ResNet model class.
    """
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
    ) -> None:
        """
        Args:
            block (Type[Union[BasicBlock, Bottleneck]]): The block type to use.
            layers (List[int]): A list with 4 numbers, specifying the
                                number of blocks in each of the 4 stages.
            num_classes (int): Number of output classes (e.g., 1000 for ImageNet).
        """
        super(ResNet, self).__init__()
        
        # This variable tracks the number of input channels for the *next* stage
        self.in_channels = 64
        
        # --- Stage 0: The "Stem" ---
        # 3 input channels (RGB), 64 output channels
        # Kernel 7x7, Stride 2, Padding 3
        self.conv1 = nn.Conv2d(
            3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # --- The 4 Residual Stages ---
        # Stage 1 (e.g., conv2_x)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        
        # Stage 2 (e.g., conv3_x)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        
        # Stage 3 (e.g., conv4_x)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        
        # Stage 4 (e.g., conv5_x)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # --- Classifier Head ---
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # --- Weight Initialization ---
        self._initialize_weights()

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        out_channels: int,
        num_blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        """
        Helper function to create a full residual stage (e.g., conv2_x).
        
        Args:
            block: The block type (BasicBlock or Bottleneck).
            out_channels: The number of *output* channels for the stage
                          (or *intermediate* channels for Bottleneck).
            num_blocks: The number of blocks to stack.
            stride: The stride for the *first* block of this stage.
        
        Returns:
            nn.Sequential: The complete stage.
        """
        downsample = None
        
        # This 'if' check is crucial. We need a downsample layer (1x1 conv)
        # on the shortcut connection if:
        # 1. The stride is > 1 (spatial dimensions are changing)
        # 2. The number of input channels doesn't match the final
        #    output channels of the block (channel dimensions are changing)
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        # Add the first block, which handles the stride and any downsampling
        layers.append(
            block(
                self.in_channels,
                out_channels,
                stride,
                downsample,
            )
        )
        
        # Update in_channels for the *next* block(s) in this stage
        self.in_channels = out_channels * block.expansion

        # Add the remaining blocks for this stage
        # These blocks have stride=1 and no downsampling
        for _ in range(1, num_blocks):
            layers.append(
                block(
                    self.in_channels,
                    out_channels,
                )
            )

        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        # Kaiming He initialization for conv layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                # Constant init for BatchNorm
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 4 Stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Classifier Head
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # Flatten the tensor
        x = self.fc(x)

        return x


# --- Factory Functions ---
# These functions make it easy to create specific ResNet models

def resnet18(num_classes: int = 1000) -> ResNet:
    """Creates a ResNet-18 model"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def resnet34(num_classes: int = 1000) -> ResNet:
    """Creates a ResNet-34 model"""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

def resnet50(num_classes: int = 1000) -> ResNet:
    """Creates a ResNet-50 model"""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

def resnet101(num_classes: int = 1000) -> ResNet:
    """Creates a ResNet-101 model"""
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)

def resnet152(num_classes: int = 1000) -> ResNet:
    """Creates a ResNet-152 model"""
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)


# --- Example Usage ---
if __name__ == "__main__":
    # Create a ResNet-50 model
    # You can change this to resnet18, resnet34, etc.
    model = resnet50(num_classes=1000)
    
    # You can print the model to see its architecture
    # print(model)

    # Create a dummy input tensor to test the forward pass
    # Batch size = 1, Channels = 3, Height = 224, Width = 224
    dummy_input = torch.randn(1, 3, 224, 224)

    # Pass the input through the model
    try:
        output = model(dummy_input)
        print(f"Successfully created a ResNet-50 model.")
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"Error during model forward pass: {e}")

    # Example with a different model (ResNet-18) and different classes
    model_18 = resnet18(num_classes=10) # e.g., for CIFAR-10
    dummy_input_cifar = torch.randn(1, 3, 32, 32) # CIFAR-10 images are 32x32
    
    try:
        output_18 = model_18(dummy_input_cifar)
        print(f"\nSuccessfully created a ResNet-18 model for 10 classes.")
        print(f"Input shape: {dummy_input_cifar.shape}")
        print(f"Output shape: {output_18.shape}")
    except Exception as e:
        print(f"Error during ResNet-18 forward pass: {e}")
