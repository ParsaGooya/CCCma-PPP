import numpy as np
from pathlib import Path

from cccma_ppp.data_modules.data import DataConfigABC
from cccma_ppp.data_modules.dataset import DatasetConfigABC
from cccma_ppp.data_modules import WeightsConfig
from cccma_ppp.preprocessing import PreprocessModuleABC



class DatasetOperator: 

    def __init__(self, config: DatasetConfigABC):
        self.config = config

    def _fit_preprocessors(
        self,
        train_years: np.ndarray | list | tuple,
        save=False,
        save_path: Path | str | None = None,
        save_name: str | None = None,
    ):
        
        if self.config.model is not None:
            selection = {
                "year": train_years,
                "lead_time": np.arange(1, self.config.num_lead_months + 1),
            }
            if self.config.model.info.coords["ensembles"] is not None:
                selection["ensembles"] = self.config.model.info.coords["ensembles"]

            self.config.model._fit_preprocessor_pipeline( 
                                selection = selection, 
                                mask = True, 
                                save = save,
                                save_path = save_path, 
                                save_name = save_name)
 

        if self.config.observation is not None:
                selection = {"year": train_years}
                if self.config.observation.info.coords["ensembles"] is not None:
                    selection["ensembles"] = self.config.observation.info.coords["ensembles"]

                self.config.observation._fit_preprocessor_pipeline( 
                    selection = selection, 
                    save = save,
                    save_path = save_path, 
                    save_name = save_name)


        if self.config.effective_condition is not None:
                if self.config.condition_method == 'static':
                    selection = {}
                else:
                    selection = {
                        "year": train_years,
                        "lead_time": np.arange(1, self.config.num_lead_months + 1),
                    }
                    if self.config.effective_condition.info.coords["ensembles"] is not None:
                        selection["ensembles"] = self.config.effective_condition.info.coords["ensembles"]
                
                self.config.effective_condition._fit_preprocessor_pipeline( 
                    selection = selection, 
                    mask = True,
                    save = save,
                    save_path = save_path, 
                    save_name = save_name)


        self.config._fitted_preprocessors = True

    def _load_fitted_preprocessors(
        self, load_dir: Path | str
    ):

        if self.config.model is not None:

            self.config.model._load_preprocessor_pipeline(load_dir)

        if self.config.observation is not None:

            self.config.observation._load_preprocessor_pipeline(load_dir)

        if self.config.effective_condition is not None:

            self.config.effective_condition._load_preprocessor_pipeline(load_dir)

        self.config._fitted_preprocessors = True

    def _add_fitted_preprocessor(self, preprocessor: PreprocessModuleABC, index=0):

        if not isinstance(preprocessor, PreprocessModuleABC):
            raise TypeError(
                f"preprocessor must be an instance of ProcessorConfig, "
                f"got {type(preprocessor)}"
            )
        assert preprocessor.fitted, "The preprocessor must be fitted"

        if self.config.model is not None:
            self.config.model.preprocessing_pipeline.add_fitted_preprocessor(
                preprocessor, index=index
            )
        if self.config.observation is not None:
            self.config.observation.preprocessing_pipeline.add_fitted_preprocessor(
                preprocessor, index=index
            )
        if self.config.effective_condition is not None:
            self.config.effective_condition.preprocessing_pipeline.add_fitted_preprocessor(
                preprocessor, index=index
            )

    def get_weights(
        self,
        config: WeightsConfig | None = None,
        save=True,
        save_path: Path | str | None = None,
        save_name: str | None = None,
    ):
        if config is None:
            config = WeightsConfig()

        if self.config.observation is not None:
            target_coords = self.config.observation.info.coords.copy()
        elif self.config.model is not None:
            target_coords = self.config.model.info.coords.copy()
        else:
            raise ValueError(
                'No model or observation data is availablle. ' \
                'Weights could not be generated')

        if "ensembles" in target_coords:
            del target_coords["ensembles"]

        from cccma_ppp.preprocessing.utils_preprocessing import Oceannanremove

        if self.config.observation is not None:
            pipeline = self.config.observation.preprocessing_pipeline
        else:
            pipeline = self.config.model.preprocessing_pipeline

        checklist = [
            isinstance(item, Oceannanremove) for item in pipeline.fitted_preprocessors
        ]

        weights = config.build_weights(
            target_coords,
            oceannanremover=pipeline.get_preprocessors("oceannanremover")
            if any(checklist)
            else None,
            save=save,
            save_path=save_path,
            save_name=save_name,
        )

        if "channels" in weights.dims:
            if self.config.observation is not None:
                error_msg = f"inconsistent variable weights {weights.channels.values} for taget variables {self.config.observation.names}"
                if not weights.channels.values == self.config.observation.names:
                    raise RuntimeError(error_msg)
            else:
                error_msg = f"inconsistent variable weights {weights.channels.values} for taget variables {self.config.model.names}"
                if not weights.channels.values == self.config.model.names:
                    raise RuntimeError(error_msg)

        return weights

    def get_input_var_metadata(self):

        metadata = dict(variables=list(), preprocessors=list())

        if self.config.effective_condition is None:
            metadata = self._update_metadata_with_dataconfig_metadata(metadata, self.config.model)

        else:
            if not self.config._using_model_data_as_condition:
                metadata = self._update_metadata_with_dataconfig_metadata(
                    metadata, self.config.model
                )
                metadata = self._update_metadata_with_dataconfig_metadata(
                    metadata, self.config.effective_condition
                )
            else:
                metadata = self._update_metadata_with_dataconfig_metadata(
                    metadata, self.config.effective_condition
                )

        return metadata

    def get_target_var_metadata(self):

        metadata = dict(variables=list(), preprocessors=list())

        if self.config.observation is None:
            if self.config.model is None:
                raise ValueError(
                'No model or observation data is availablle. ' \
                'target variable metadata could not be generated')
            
            metadata = self._update_metadata_with_dataconfig_metadata(metadata, 
                                                                      self.config.model
                                                                      )
        else:
            metadata = self._update_metadata_with_dataconfig_metadata(
                metadata, self.config.observation
            )

        return metadata

    def _update_metadata_with_dataconfig_metadata(
            self, metadata: dict, dataconfig: DataConfigABC
        ):
            preprocessor_names = [
                processor[0] for processor in dataconfig.preprocessing_pipeline.pipeline
            ]
            for var in dataconfig.names:
                metadata["variables"].append(var)
                metadata["preprocessors"].append(preprocessor_names)

            return metadata


      
