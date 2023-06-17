import eel

from inpaintGAN.Config import Config
from inpaintGAN.edgeConnect import EdgeConnect
from PIL import Image
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
def download_file(url):
    decoded_data = base64.b64decode(url)
    img_file = open('web/inference/image.jpeg', 'wb')
    img_file.write(decoded_data)
    img_file.close()

if __name__ == '__main__':
    eel.start("index.html")
