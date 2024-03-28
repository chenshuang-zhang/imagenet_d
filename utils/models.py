from torchvision.datasets import ImageFolder
from torchvision import models, transforms, datasets

model_dict = { 
        'vgg19': [models.vgg19_bn, models.VGG19_BN_Weights.IMAGENET1K_V1],
        'vit_b_16': [models.vit_b_16, models.ViT_B_16_Weights.IMAGENET1K_V1],
}

def load_model(model_name):
    weights = model_dict[model_name][1]
    model = model_dict[model_name][0](weights=weights)
    test_transform = weights.transforms()
    return model, test_transform