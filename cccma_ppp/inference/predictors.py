from cccma_ppp.core.selectors import PredictorSelector
import dataclasses





@PredictorSelector.register("cvae")
@dataclasses.dataclass
class cVAEPredictorConfig:
    """
    Configuration for conditional variational autoencoder (cVAE) predictor.

    Parameters
    ----------
    nu : cVAEModelSelector or None
        Model configuration selector.
    min_posterior_variance : float or None
        Minimum allowed posterior variance.
    prior_flow_config : NormalizedFlowConfig or None
        Configuration for optional prior flow.
    combined_CGCN_weight : float or None
        Weight for auxiliary CGCN loss.
    load_dir : str or None
        Path to checkpoint for loading configuration.
    """

    num_latent_samples: int = 1
    infer_prior_from_training: bool = False


    def __post_init__(self):
        """
        Validate and initialize configuration.

        Raises
        ------
        ValueError
            If neither `ModelConfig` nor `load_dir` is provided.
        AssertionError
            If `combined_CGCN_weight` is not in [0, 1].
        """

        if self.load_dir is None:
            if self.ModelConfig is None:
                raise ValueError("provide loading dir or model configurations")

        else:
            self._load_from_checkpoint(self.load_dir)
            warnings.warn(
                f"Model and prior flow config overwritten by the saved module: \n {self.load_dir}"
            )

        if self.combined_CGCN_weight is None:
            self.combined_CGCN_weight = 0

        self.model_config = self.ModelConfig.get_model_config()

        self.latent_size = self.model_config.latent_size

        if self.prior_flow_config is not None:
            self.condition_dependant_flow = self.model_config.condition_dependant_latent

            if self.condition_dependant_flow:
                self.model_config._resolve_flow_settings(self.condition_dependant_flow)

        assert 0 <= self.combined_CGCN_weight <= 1, (
            "CGCN weight should be between [0,1]"
        )

    def build(
        self,
        input_shape: np.ndarray,
        output_shape: np.ndarray | None = None,
        added_features_dim: int = None,
    ):
        """
        Construct cVAE module instance.

        Parameters
        ----------
        input_shape : np.ndarray
            Input tensor shape.
        output_shape : np.ndarray or None, optional
            Output tensor shape.
        added_features_dim : int, optional
            Additional feature dimension.

        Returns
        -------
        cVAE
            Initialized cVAE module.
        """

        return cVAE(
            config=self,
            input_shape=input_shape,
            output_shape=output_shape,
            added_features_dim=added_features_dim,
        )