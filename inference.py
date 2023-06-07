import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision.io import read_image


def inference(image_path: str) -> torch.tensor:
    ''' run a model on given image '''
    model_file = 'final_swinunetr.pt'
    loaded_model = torch.load(model_file, map_location=torch.device('cpu'))
    loaded_model.eval()

    image = image_preprocessing(image_path)
    model_out = loaded_model(image.unsqueeze(0))
    return model_out[0].detach().permute(1, 2, 0) > 0.5


def image_preprocessing(image_path: str) -> torch.tensor:
    ''' preprocess image '''
    image = read_image(image_path)
    resize = torchvision.transforms.Resize(224, antialias=True)
    image = resize(image)
    image = image / 255
    return image


if __name__ == "__main__":
    out = inference('examples/bjorke_1.png').to(dtype=torch.float32)
    plt.imshow(out)
    plt.show()
