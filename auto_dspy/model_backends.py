from abc import ABC, abstractmethod
from pydantic_settings import BaseSettings
from transformers import pipeline
import timm
import torch


class BackendConfig(BaseSettings):
    pass


class TransformersBackendConfig(BackendConfig):
    task: str
    model: str


class TimmBackendConfig(BackendConfig):
    model: str


class ModelBackend(ABC):
    @abstractmethod
    def __load_model(self, config: BackendConfig):
        pass

    @abstractmethod
    def __initialize_model(self):
        pass

    @abstractmethod
    def run(self, input: str):
        pass


class TransformersBackend(ModelBackend):
    def __init__(self, config: TransformersBackendConfig) -> None:
        self.model = self.__load_model(config)

    def __load_model(self, config: TransformersBackendConfig):
        return pipeline(task=config.task, model=config.model)

    def run(self, input: str):
        return self.model(input)


class TimmBackend(ModelBackend):
    def __init__(self, config: TimmBackendConfig) -> None:
        self.model = self.__load_model(config)

    def __load_model(self, config: TimmBackendConfig):
        if config.model not in timm.list_models(pretrained=True):
            raise Exception(f"model {config.model} for timm is not pretrained")

        return timm.create_model(
            model_name=config.model, pretrained=True, checkpoint_path="weights/"
        ).eval()

    def run(self, input):
        transform = timm.data.create_transform(
            **timm.data.resolve_data_config(self.model.pretrained_cfg)
        )
        image_tensor = transform(input)
        output = self.model(image_tensor.unsqueeze(0))
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        values, indices = torch.topk(probabilities, 1)
        return values, indices
