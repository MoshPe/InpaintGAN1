import eel

from inpaintGAN.Config import Config
from inpaintGAN.edgeConnect import EdgeConnect
from PIL import Image, ImageTk
from torchvision.transforms import ToPILImage
import base64

modelConfig = {}

eel.init("web", allowed_extensions=['.js', '.html', '.png', '.txt'])


@eel.expose
def train_model():
    edgeConfig = Config()
    edgeConnect = EdgeConnect(edgeConfig)
    edgeConnect.train(eel.setMetrics, eel.addLog)


@eel.expose
def save_model_config(x):
    for key in x:
        try:
            modelConfig[key] = int(x[key])
        except:
            try:
                modelConfig[key] = float(x[key])
            except:
                modelConfig[key] = x[key]
    print(modelConfig)


@eel.expose
def test_model():
    edgeConfig = Config()
    edgeConfig.MASK = 6
    edgeConfig.MODE = 4
    edgeConnect = EdgeConnect(edgeConfig)
    edgeConnect.load()
    img = edgeConnect.fill_image(256)
    tensor_to_pil = ToPILImage()
    pil_image = tensor_to_pil(img.squeeze())
    pil_image.save("web/inference/test.jpeg")  # Replace "image.jpg" with the desired file path and extension


@eel.expose
def download_image_file(url):
    decoded_data = base64.b64decode(url)
    img_file = open('web/inference/image.jpeg', 'wb')
    img_file.write(decoded_data)
    img_file.close()


@eel.expose
def download_mask_file(url):
    decoded_data = base64.b64decode(url)
    img_file = open('web/inference/mask.jpeg', 'wb')
    img_file.write(decoded_data)
    img_file.close()


if __name__ == '__main__':
    eel.start("index.html")
