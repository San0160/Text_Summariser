from Text_summariser.config.configuration import configurationManager
from Text_summariser.utils.common import download_model


def main():
    config = configurationManager().get_model_evaluation_config()
    download_model(config)


if __name__ == "__main__":
    main()