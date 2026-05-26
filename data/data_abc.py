import abc
# from pydantic import BaseModel, model_validator




class XarrayDatasetConfigABC(abc.ABC):   ###the ABC or Abstract Base Class as a parent indicates Any subclass MUST implement .build() with this signature. This is a common pattern for defining interfaces in Python.
    @abc.abstractmethod
    def build(
        self):
        ...



class XarrayDatasetABC(abc.ABC):
    @abc.abstractmethod
    def __getitem__(self, index):
        ...
    
    @abc.abstractmethod
    def __len__(self):
        ...


    # @property
    # @abc.abstractmethod
    # def to_device():
    #     pass






# @dataclass
# class datadict(BaseModel):
#     paths: list[str]
#     names: list[str]
#     preprocessing_pipeline :  #list[tuple[str, dict]]
#     concat_dim : str = 'year'

#     @model_validator(mode = 'after')
#     def check_requirements(self):    
#         _check_data(self.paths, self.data, self.concat_dim)



# class conditiondatadict(BaseModel):
#     paths: list[str]
#     names: list[str]
#     condition_method: str
#     concat_dim : str = 'year'

#     @model_validator(mode = 'after')
#     def check_requirements(self):
#         _check_data(self.paths, self.data, self.concat_dim)
#         assert self.condition_method in ['ensemble_mean' , 'climatology' , 'cross_ensemble'], f'{self.condition_method} is not a valid conditioning method.'

