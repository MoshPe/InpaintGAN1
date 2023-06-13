from Config import Config
from edgeConnect import EdgeConnect


if __name__ == '__main__':
    edgeConfig = Config()

    edgeConnect = EdgeConnect(edgeConfig)
    edgeConnect.train()