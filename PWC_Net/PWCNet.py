#!/usr/bin/env python

import torch
import math

try:
    from .correlation import correlation # the custom cost volume layer
except:
    import sys
    sys.path.insert(0, './correlation')
    import correlation # you should consider upgrading python
# end

##########################################################

backwarp_tenGrid = {}
backwarp_tenPartial = {}

def backwarp(tenInput, tenFlow):
    if str(tenFlow.shape) not in backwarp_tenGrid:
        tenHor = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, -1).repeat(1, 1, tenFlow.shape[2], 1)
        tenVer = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, -1, 1).repeat(1, 1, 1, tenFlow.shape[3])

        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([ tenHor, tenVer ], 1).cuda()
    # end

    if str(tenFlow.shape) not in backwarp_tenPartial:
        backwarp_tenPartial[str(tenFlow.shape)] = tenFlow.new_ones([ tenFlow.shape[0], 1, tenFlow.shape[2], tenFlow.shape[3] ])
    # end

    tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] * (2.0 / (tenInput.shape[3] - 1.0)), tenFlow[:, 1:2, :, :] * (2.0 / (tenInput.shape[2] - 1.0)) ], 1)
    tenInput = torch.cat([ tenInput, backwarp_tenPartial[str(tenFlow.shape)] ], 1)

    tenOutput = torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=True)

    tenMask = tenOutput[:, -1:, :, :]; tenMask[tenMask > 0.999] = 1.0; tenMask[tenMask < 1.0] = 0.0

    return tenOutput[:, :-1, :, :] * tenMask
# end

##########################################################

class PWCDCNet(torch.nn.Module):
    def __init__(self, strModel='default', ckpt_path=None):
        super().__init__()

        class Extractor(torch.nn.Module):
            def __init__(self):
                super().__init__()

                self.netOne = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netTwo = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netThr = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netFou = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netFiv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netSix = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=128, out_channels=196, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )
            # end

            def forward(self, tenInput):
                # print(f"🔍 Extractor输入: {tenInput.shape}, 值范围: [{tenInput.min():.6f}, {tenInput.max():.6f}]")
                
                tenOne = self.netOne(tenInput)
                # print(f"🔍 netOne后: {tenOne.shape}, 值范围: [{tenOne.min():.6f}, {tenOne.max():.6f}]")
                
                tenTwo = self.netTwo(tenOne)
                # print(f"🔍 netTwo后: {tenTwo.shape}, 值范围: [{tenTwo.min():.6f}, {tenTwo.max():.6f}]")
                
                tenThr = self.netThr(tenTwo)
                # print(f"🔍 netThr后: {tenThr.shape}, 值范围: [{tenThr.min():.6f}, {tenThr.max():.6f}]")
                
                tenFou = self.netFou(tenThr)
                # print(f"🔍 netFou后: {tenFou.shape}, 值范围: [{tenFou.min():.6f}, {tenFou.max():.6f}]")
                
                tenFiv = self.netFiv(tenFou)
                # print(f"🔍 netFiv后: {tenFiv.shape}, 值范围: [{tenFiv.min():.6f}, {tenFiv.max():.6f}]")
                # print(f"🔍 netFiv数据值：")
                # print(tenFiv.min(), tenFiv.max())
                # print(tenFiv)
                tenSix = self.netSix(tenFiv)
                # print(f"🔍 netSix后: {tenSix.shape}, 值范围: [{tenSix.min():.6f}, {tenSix.max():.6f}]")

                return [ tenOne, tenTwo, tenThr, tenFou, tenFiv, tenSix ]
            # end
        # end

        class Decoder(torch.nn.Module):
            def __init__(self, intLevel):
                super().__init__()

                intPrevious = [ None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None ][intLevel + 1]
                intCurrent = [ None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None ][intLevel + 0]

                if intLevel < 6: self.netUpflow = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)
                if intLevel < 6: self.netUpfeat = torch.nn.ConvTranspose2d(in_channels=intPrevious + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=4, stride=2, padding=1)
                if intLevel < 6: self.fltBackwarp = [ None, None, None, 5.0, 2.5, 1.25, 0.625, None ][intLevel + 1]

                self.netOne = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netTwo = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netThr = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128, out_channels=96, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netFou = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netFiv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netSix = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=3, stride=1, padding=1)
                )
            # end

            def forward(self, tenOne, tenTwo, objPrevious):
                tenFlow = None
                tenFeat = None

                if objPrevious is None:
                    tenFlow = None
                    tenFeat = None

                    tenVolume = torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(tenOne=tenOne, tenTwo=tenTwo), negative_slope=0.1, inplace=False)

                    tenFeat = torch.cat([ tenVolume ], 1)

                elif objPrevious is not None:
                    tenFlow = self.netUpflow(objPrevious['tenFlow'])
                    tenFeat = self.netUpfeat(objPrevious['tenFeat'])

                    tenVolume = torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(tenOne=tenOne, tenTwo=backwarp(tenInput=tenTwo, tenFlow=tenFlow * self.fltBackwarp)), negative_slope=0.1, inplace=False)

                    tenFeat = torch.cat([ tenVolume, tenOne, tenFlow, tenFeat ], 1)

                # end

                tenFeat = torch.cat([ self.netOne(tenFeat), tenFeat ], 1)
                # print(f"🔍 netOne后 tenFeat shape: {tenFeat.shape}")
                tenFeat = torch.cat([ self.netTwo(tenFeat), tenFeat ], 1)
                # print(f"🔍 netTwo后 tenFeat shape: {tenFeat.shape}")
                tenFeat = torch.cat([ self.netThr(tenFeat), tenFeat ], 1)
                # print(f"🔍 netThr后 tenFeat shape: {tenFeat.shape}")
                tenFeat = torch.cat([ self.netFou(tenFeat), tenFeat ], 1)
                # print(f"🔍 netFou后 tenFeat shape: {tenFeat.shape}")
                # print(f"🔍 准备调用netFiv，输入通道数: {tenFeat.shape[1]}")
                # print(f"🔍 netFiv期望通道数: {self.netFiv[0].in_channels}")
                tenFeat = torch.cat([ self.netFiv(tenFeat), tenFeat ], 1)

                tenFlow = self.netSix(tenFeat)

                return {
                    'tenFlow': tenFlow,
                    'tenFeat': tenFeat
                }
            # end
        # end

        class Refiner(torch.nn.Module):
            def __init__(self):
                super().__init__()

                self.netMain = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=81 + 32 + 2 + 2 + 128 + 128 + 96 + 64 + 32, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8, dilation=8),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=16, dilation=16),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1)
                )
            # end

            def forward(self, tenInput):
                return self.netMain(tenInput)
            # end
        # end

        self.netExtractor = Extractor()

        self.netTwo = Decoder(2)
        self.netThr = Decoder(3)
        self.netFou = Decoder(4)
        self.netFiv = Decoder(5)
        self.netSix = Decoder(6)

        self.netRefiner = Refiner()

        # Load weights
        if ckpt_path is not None:
            # Load from local checkpoint
            try:
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                # Convert key names: 'module' -> 'net'
                converted_state_dict = {strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in state_dict.items()}
                self.load_state_dict(converted_state_dict, strict=False)
                print(f"Successfully loaded PWC-Net weights from {ckpt_path}")
            except Exception as e:
                print(f"Warning: Failed to load local weights from {ckpt_path}: {e}")
                print("Falling back to pre-trained weights from torch.hub")
                # Fallback to pre-trained weights
                self.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/github/pytorch-pwc/network-' + strModel + '.pytorch', file_name='pwc-' + strModel).items() })
        else:
            # Load pre-trained weights from torch.hub
            self.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/github/pytorch-pwc/network-' + strModel + '.pytorch', file_name='pwc-' + strModel).items() })
    # end

    def forward(self, tenOne, tenTwo):
        # print(f"🔍 PWC-Net 输入检查:")
        # print(f"  tenOne输入: {tenOne.shape}, dtype: {tenOne.dtype}, device: {tenOne.device}")
        # print(f"  tenTwo输入: {tenTwo.shape}, dtype: {tenTwo.dtype}, device: {tenTwo.device}")
        # print(f"  tenOne值范围: [{tenOne.min():.6f}, {tenOne.max():.6f}]")
        # print(f"  tenTwo值范围: [{tenTwo.min():.6f}, {tenTwo.max():.6f}]")
        
        tenOne = self.netExtractor(tenOne)
        tenTwo = self.netExtractor(tenTwo)

        # print(f"🔍 PWC-Net forward开始:")
        # print(f"  tenOne长度: {len(tenOne)}")
        # for i, feat in enumerate(tenOne):
        #     print(f"  tenOne[{i}]: {feat.shape}")
        # print(f"  tenTwo长度: {len(tenTwo)}")
        # for i, feat in enumerate(tenTwo):
        #     print(f"  tenTwo[{i}]: {feat.shape}")
        
        try:
            # print(f"🔍 执行netSix...")
            # print('##########################################################')
            # print(tenOne[-1].shape, tenTwo[-1].shape)
            # print('tenOne数据值：')
            # print(tenTwo[-1].min(), tenTwo[-1].max())
            # print(tenOne[-1])
            objEstimate = self.netSix(tenOne[-1], tenTwo[-1], None)
            # print(f"✅ netSix完成: {objEstimate['tenFlow'].shape if 'tenFlow' in objEstimate else 'No tenFlow'}")
        except Exception as e:
            # print(f"❌ netSix失败: {e}")
            raise e
            
        try:
            # print(f"🔍 执行netFiv...")
            objEstimate = self.netFiv(tenOne[-2], tenTwo[-2], objEstimate)
            # print(f"✅ netFiv完成: {objEstimate['tenFlow'].shape if 'tenFlow' in objEstimate else 'No tenFlow'}")
        except Exception as e:
            # print(f"❌ netFiv失败: {e}")
            raise e
            
        try:
            # print(f"🔍 执行netFou...")
            objEstimate = self.netFou(tenOne[-3], tenTwo[-3], objEstimate)
            # print(f"✅ netFou完成: {objEstimate['tenFlow'].shape if 'tenFlow' in objEstimate else 'No tenFlow'}")
        except Exception as e:
            # print(f"❌ netFou失败: {e}")
            raise e
            
        try:
            # print(f"🔍 执行netThr...")
            objEstimate = self.netThr(tenOne[-4], tenTwo[-4], objEstimate)
            # print(f"✅ netThr完成: {objEstimate['tenFlow'].shape if 'tenFlow' in objEstimate else 'No tenFlow'}")
        except Exception as e:
            # print(f"❌ netThr失败: {e}")
            raise e
            
        try:
            # print(f"🔍 执行netTwo...")
            objEstimate = self.netTwo(tenOne[-5], tenTwo[-5], objEstimate)
            # print(f"✅ netTwo完成: {objEstimate['tenFlow'].shape if 'tenFlow' in objEstimate else 'No tenFlow'}")
        except Exception as e:
            # print(f"❌ netTwo失败: {e}")
            raise e

        return (objEstimate['tenFlow'] + self.netRefiner(objEstimate['tenFeat'])) * 20.0
    # end
# end

##########################################################

def estimate(tenOne, tenTwo, strModel='default'):
    """
    Estimate optical flow between two images using PWCDCNet.
    
    Args:
        tenOne: First input tensor (B, C, H, W)
        tenTwo: Second input tensor (B, C, H, W)
        strModel: Model variant ('default' or 'chairs-things')
    
    Returns:
        Optical flow tensor
    """
    netNetwork = PWCDCNet(strModel).cuda().train(False)

    assert(tenOne.shape[1] == tenTwo.shape[1])
    assert(tenOne.shape[2] == tenTwo.shape[2])

    intWidth = tenOne.shape[2]
    intHeight = tenOne.shape[1]

    assert(intWidth == 1024) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
    assert(intHeight == 436) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

    tenPreprocessedOne = tenOne.cuda().view(1, 3, intHeight, intWidth)
    tenPreprocessedTwo = tenTwo.cuda().view(1, 3, intHeight, intWidth)

    intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
    intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

    tenPreprocessedOne = torch.nn.functional.interpolate(input=tenPreprocessedOne, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
    tenPreprocessedTwo = torch.nn.functional.interpolate(input=tenPreprocessedTwo, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

    tenFlow = torch.nn.functional.interpolate(input=netNetwork(tenPreprocessedOne, tenPreprocessedTwo), size=(intHeight, intWidth), mode='bilinear', align_corners=False)

    tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
    tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

    return tenFlow[0, :, :, :].cpu()
# end
