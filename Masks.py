import numpy as np


class Masks():
    def __init__(self, imageSize, maskSize, image):
        self.imageSize = imageSize
        self.maskSize = maskSize
        self.image = image
    def centralRegion(self):
        middle = (self.imageSize - self.imageSize)/2
        maskImage = self.image.clone()
        maskImage[:, middle: middle + self.maskSize, middle: middle + self.maskSize] = 1
        return maskImage, middle

    def randomBlock(self):
        x = np.random.randint(0, self.image - self.maskSize)
        y = np.random.randint(0, self.image - self.maskSize)
        maskPart = self.image[:,y:y+self.maskSize,x:x+self.maskSize]
        maskImage = self.image.clone()
        maskImage[:, y:y + self.maskSize, x:x + self.maskSize] = 1
        return maskImage, maskPart

    def randomRegion(self, samples, maskArea, globalRandomPattern):
        # samples: torch.Tensor, img_size: int, mask_area: float,  global_random_pattern: torch.Tensor
        while True:
            x = np.random.randint(0, globalRandomPattern.size()[0] - self.imageSize)
            y = np.random.randint(0, globalRandomPattern.size()[0] - self.imageSize)
            mask = globalRandomPattern[x: x + self.imageSize, y: y + self.imageSize]
            patterMaskArea = mask.float().mean().item()
            # If mask area is within +/- 25% of desired mask area, break and continue with line 364
            if maskArea / 1.25 < patterMaskArea < maskArea * 1.25:
                break
        masked_samples = samples.clone()
        masked_samples[:, 0, mask] = 2 * 117.0 / 255.0 - 1.0
        masked_samples[:, 1, mask] = 2 * 104.0 / 255.0 - 1.0
        masked_samples[:, 2, mask] = 2 * 123.0 / 255.0 - 1.0
        return masked_samples, mask