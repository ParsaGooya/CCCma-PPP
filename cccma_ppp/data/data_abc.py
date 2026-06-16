import abc


class DatasetOperatorABC(abc.ABC):
    @abc.abstractmethod
    def _fit_preprocessors(self):
        pass

    @abc.abstractmethod
    def _load_fitted_preprocessors(self):
        pass
    
    @abc.abstractmethod
    def get_weights(self):
        pass

    @abc.abstractmethod
    def get_input_var_metadata(self):
        pass
    
    @abc.abstractmethod
    def get_target_var_metadata(self):
        pass
    
    @abc.abstractmethod
    def build_dataset(self):
        pass

class DatasetConfigABC(abc.ABC):
    @abc.abstractmethod
    def build_operator(self) -> DatasetOperatorABC:
        pass

    

class DataConfigABC(abc.ABC):

    def __init__(self):
        if not hasattr(self, "preprocessing_pipeline"):
            raise AttributeError(
                f"{type(self).__name__} must define preprocessing_pipeline"
            )
        
        self.preprocessing_pipeline.set_name(self.TYPE)


    @property
    @abc.abstractmethod
    def TYPE(self) -> str:
        pass
    
    @classmethod
    @abc.abstractmethod
    def _allowed_dims(cls) -> frozenset[str]:
        pass
    
    @classmethod
    @abc.abstractmethod
    def _required_dims(cls) -> frozenset[str]:
        pass
