import torch
import numpy as np
import xarray as xr
from typing import Dict, Optional, Union
from cccma_ppp.architectures.models_abc import DeterministicRequest
from cccma_ppp.core.models.deterministic import deterministicOutput
from cccma_ppp.preprocessing.utils_preprocessing import Standardizer


class InverseTransformationMapper:
    def __init__(self, standardizers: Dict[int, Standardizer]):
        # Store actual Standardizer objects instead of stats
        self.standardizers = standardizers
        if len(standardizers) != 12:
            raise ValueError("Must have 12 standardizers (one per month)")

    def apply_inverse_transformation(
        self,
        output: torch.Tensor,
        init_month: Union[int, torch.Tensor],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        # Inverse transformation using Standardizer.inverse_transform()
        if output.dim() == 3:
            output = output.unsqueeze(0)
            squeeze_output = True
        elif output.dim() == 4:
            squeeze_output = False
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {output.dim()}D")

        batch_size = output.shape[0]

        if isinstance(init_month, torch.Tensor):
            if init_month.shape[0] != batch_size:
                raise ValueError("init_month batch size mismatch")
            init_months = init_month.cpu().numpy().astype(int)
        else:
            init_months = np.full(batch_size, init_month, dtype=int)

        denormalized = torch.zeros_like(output)
        for i, month in enumerate(init_months):
            if not 0 <= month <= 11:
                raise ValueError(f"init_month must be 0-11, got {month}")

            # Convert tensor to numpy, wrap in xarray, denormalize using Standardizer
            output_np = output[i].cpu().numpy()
            output_xr = xr.DataArray(output_np, dims=["var", "height", "width"])

            # Use Standardizer.inverse_transform() (line 10550 in src)
            denormalized_xr = self.standardizers[int(month)].inverse_transform(
                output_xr
            )
            denormalized[i] = torch.from_numpy(np.array(denormalized_xr.data)).float()

        if squeeze_output:
            denormalized = denormalized.squeeze(0)

        return denormalized

    def process_deterministic_output(
        self,
        det_output: deterministicOutput,
        init_month: Union[int, torch.Tensor],
    ) -> deterministicOutput:
        denormalized = self.apply_inverse_transformation(det_output.output, init_month)
        return deterministicOutput(output=denormalized)

    def process_deterministic_request(
        self,
        model,
        request: DeterministicRequest,
        init_month: Union[int, torch.Tensor],
    ) -> deterministicOutput:
        model_output = model.forward(request)
        return self.process_deterministic_output(model_output, init_month)


def create_standardizers_per_month():
    # Create 12 Standardizer instances, one per month (lines 10407-10456 in src)
    standardizers = {}

    for month in range(12):
        standardizer = Standardizer(dims=[], frequency=None)

        # Create xarray dataset with multiple variables (realistic scenario)
        monthly_data = np.random.randn(100, 3, 64, 64) + month * 0.3
        monthly_data_xr = xr.DataArray(
            monthly_data,
            dims=["time", "var", "height", "width"],
            coords={"time": np.arange(100)},
        )

        # Fit: computes mean/std (lines 10478-10490 in src)
        standardizer.fit(monthly_data_xr)
        standardizers[month] = standardizer

    return standardizers


def main():
    print("Creating 12 Standardizer instances (one per month)...")
    standardizers = create_standardizers_per_month()
    mapper = InverseTransformationMapper(standardizers)

    # ===== FORWARD NORMALIZATION =====
    print("\n[Forward Normalization - using Standardizer.transform()]")
    original_sample = np.random.randn(3, 64, 64).astype(np.float32)
    original_sample_xr = xr.DataArray(original_sample, dims=["var", "height", "width"])

    month = 5
    standardizer = standardizers[month]

    # Use Standardizer.transform() (line 10521 in src: (data - mean) / std)
    normalized_sample_xr = standardizer.transform(original_sample_xr)
    normalized_sample_np = np.array(normalized_sample_xr.data)
    normalized_sample = torch.from_numpy(normalized_sample_np).float()
    print(f"  Original shape: {original_sample.shape}")
    print(f"  Normalized shape: {normalized_sample.shape}")
    print(
        f"  Original mean: {original_sample.mean():.4f}, std: {original_sample.std():.4f}"
    )
    print(
        f"  Normalized mean: {normalized_sample.mean():.4f}, std: {normalized_sample.std():.4f}"
    )

    # ===== INVERSE TRANSFORMATION (using mapper) =====
    print("\n[Inverse Transformation - using InverseTransformationMapper]")
    denormalized_sample = mapper.apply_inverse_transformation(
        normalized_sample, init_month=month
    )
    print(f"  Denormalized shape: {denormalized_sample.shape}")
    print(
        f"  Denormalized mean: {denormalized_sample.mean():.4f}, std: {denormalized_sample.std():.4f}"
    )

    error_mapper = np.abs(original_sample - denormalized_sample.numpy()).max()
    print(f"  Reconstruction error (mapper): {error_mapper:.2e}")

    # ===== ROUNDTRIP VERIFICATION =====
    print("\n[Roundtrip Verification - Standardizer.transform() → inverse_transform()]")
    test_month = 7
    original_data = np.random.randn(3, 64, 64).astype(np.float32)
    original_data_xr = xr.DataArray(original_data, dims=["var", "height", "width"])

    standardizer = standardizers[test_month]

    # Forward: Standardizer.transform() (line 10521 in src)
    normalized_xr = standardizer.transform(original_data_xr)
    print(f"  Original mean: {float(original_data_xr.mean()):.4f}")
    print(f"  Normalized mean: {float(normalized_xr.mean()):.4f}")

    # Inverse: Standardizer.inverse_transform() (line 10550 in src)
    recovered_xr = standardizer.inverse_transform(normalized_xr)
    recovered_np = np.array(recovered_xr.data)

    error_standardizer = np.abs(original_data - recovered_np).max()
    print(f"  Recovered mean: {recovered_np.mean():.4f}")
    print(f"  Reconstruction error (Standardizer): {error_standardizer:.2e}")

    # ===== BATCH PROCESSING WITH MIXED MONTHS =====
    print("\n[Batch Processing - Mixed Initialization Months]")
    batch_size = 12
    init_months = torch.tensor(list(range(12)))  # One sample per month
    batch_normalized = torch.randn(batch_size, 3, 64, 64)

    batch_denormalized = mapper.apply_inverse_transformation(
        batch_normalized, init_month=init_months
    )
    print(f"  Batch size: {batch_size}")
    print(f"  Normalized batch shape: {batch_normalized.shape}")
    print(f"  Denormalized batch shape: {batch_denormalized.shape}")
    print("  Each sample denormalized with its respective month's Standardizer")

    # ===== CCCMA_PPP DETERMINISTIC REQUEST INTEGRATION =====
    print("\n[DeterministicRequest Integration]")

    class MockDeterministicModel(torch.nn.Module):
        def forward(self, request: DeterministicRequest) -> deterministicOutput:
            # Model outputs normalized predictions
            output = request.input * 0.95 + 0.05
            return deterministicOutput(output=output)

    model = MockDeterministicModel()
    model.eval()

    input_tensor = torch.randn(8, 3, 64, 64)
    request = DeterministicRequest(input=input_tensor)

    with torch.no_grad():
        model_output = model.forward(request)

    print(f"  Model input shape: {input_tensor.shape}")
    print(f"  Model output shape: {model_output.output.shape}")

    # Denormalize using mapper with mixed months
    test_months = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    denormalized_output = mapper.process_deterministic_output(
        model_output, init_month=test_months
    )
    print(f"  Denormalized output shape: {denormalized_output.output.shape}")
    print(f"  Months used: {list(test_months.numpy())}")

    # Direct request processing
    denormalized_direct = mapper.process_deterministic_request(
        model, request, init_month=5
    )
    print(
        f"  Direct request processing output shape: {denormalized_direct.output.shape}"
    )

    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
