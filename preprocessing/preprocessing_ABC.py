import abc
# from pydantic import BaseModel, model_validator




# class pipelinedict(BaseModel):
#     name: list[str]
#     args : dict = {}




class PreprocessModuleABC(abc.ABC):
    @abc.abstractmethod 
    def fit(self, data):
        ...
    @abc.abstractmethod
    def transform(self, data, **kwargs):
        ...
    @abc.abstractmethod
    def inverse_transform(self, data, **kwargs):
        ...