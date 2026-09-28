import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional

from cccma_ppp.configs import realization_dim

# Get init_time_dim and lead_time_dim from config, with fallbacks
try:
    from cccma_ppp.data_modules.data.data_abc import init_time_dim, lead_time_dim
except ImportError:
    init_time_dim = "init_time"
    lead_time_dim = "lead_time"


def find_time_dimensions(ds: xr.Dataset) -> tuple:
    """
    Find initialization and lead time dimensions in dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to search

    Returns
    -------
    tuple
        (init_time_dim_found, lead_time_dim_found)
    """
    # Common names for these dimensions
    init_time_names = ["init_time", "initialization_time", "time", "start_time"]
    lead_time_names = ["lead_time", "forecast_time", "step"]

    init_time_found = None
    lead_time_found = None

    for name in init_time_names:
        if name in ds.dims:
            init_time_found = name
            break

    for name in lead_time_names:
        if name in ds.dims:
            lead_time_found = name
            break

    return init_time_found, lead_time_found


def find_realization_dimension(ds: xr.Dataset) -> Optional[str]:
    """
    Find realization/ensemble dimension in dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to search

    Returns
    -------
    str or None
        Name of realization dimension, or None if not found
    """
    # Common names for realization/ensemble dimension
    realization_names = [
        "ensembles",
        "realization",
        "member",
        "ensemble",
        "output_samples",
        "latent_samples",
        "run",
    ]

    for name in realization_names:
        if name in ds.dims:
            return name

    return None


def find_spatial_dimensions(var: xr.DataArray) -> list:
    """
    Find spatial dimensions in a variable.

    Parameters
    ----------
    var : xr.DataArray
        Variable to search

    Returns
    -------
    list
        Names of spatial dimensions (typically lat/lon or x/y)
    """
    spatial_names = ["lat", "latitude", "y", "north", "lon", "longitude", "x", "east"]

    spatial_dims = []
    for dim in var.dims:
        if any(s in dim.lower() for s in spatial_names):
            spatial_dims.append(dim)

    return spatial_dims


def print_realization_info(ds: xr.Dataset, verbose: bool = True) -> dict:
    """
    Print information about the realization dimension in a dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to inspect
    verbose : bool
        Whether to print output

    Returns
    -------
    dict
        Dictionary containing realization info
    """
    info = {
        "exists": False,
        "n_members": 0,
        "member_values": None,
        "variables": [],
        "dtype": None,
        "dim_name": None,
    }

    if verbose:
        print("=" * 70)
        print("REALIZATION DIMENSION INFORMATION")
        print("=" * 70)

    # Find realization dimension
    real_dim = find_realization_dimension(ds)

    if real_dim is None:
        if verbose:
            print("\nRealization dimension not found")
            print(f"  Configured name: '{realization_dim}'")
            print(f"  Available dimensions: {list(ds.dims)}")
            print("=" * 70)
        return info

    info["exists"] = True
    info["dim_name"] = real_dim
    info["n_members"] = ds.sizes[real_dim]
    info["member_values"] = ds[real_dim].values if real_dim in ds.coords else None
    info["dtype"] = ds[real_dim].dtype if real_dim in ds.coords else type(None)

    if verbose:
        print(f"\nRealization Dimension: '{real_dim}'")
        print(f"  Number of members: {info['n_members']}")
        if info["member_values"] is not None:
            print(f"  Member values: {info['member_values']}")
        print(f"  Member data type: {info['dtype']}")

        # Print coordinate information
        if real_dim in ds.coords:
            print("\n  Coordinate info:")
            print(f"    Shape: {ds[real_dim].shape}")
            print(f"    Dims: {ds[real_dim].dims}")
            print(f"    Attrs: {dict(ds[real_dim].attrs) or 'None'}")

        # Print variables with realization dimension
        print(f"\n  Variables with '{real_dim}' dimension:")
        for var_name, var in ds.data_vars.items():
            if real_dim in var.dims:
                print(f"    - {var_name}: shape={var.shape}, dtype={var.dtype}")
                info["variables"].append(var_name)

        # Print other dimensions for context
        print("\n  Other dimensions:")
        for dim in ds.dims:
            if dim != real_dim:
                print(f"    - {dim}: {ds.sizes[dim]}")

        print("\n" + "=" * 70)

    return info


def plot_first_ensemble_member(
    ds: xr.Dataset,
    var_name: str,
    real_dim: str,
    init_time_idx: int = 0,
    lead_time_idx: int = 0,
    figsize: tuple = (12, 6),
) -> tuple:
    """
    Plot the first ensemble member of a variable.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset
    var_name : str
        Name of the variable to plot
    real_dim : str
        Name of realization dimension
    init_time_idx : int
        Index of initialization time to plot
    lead_time_idx : int
        Index of lead time to plot
    figsize : tuple
        Figure size (width, height)

    Returns
    -------
    tuple
        (fig, ax) matplotlib objects
    """
    print(f"\nPlotting first ensemble member of '{var_name}'...")

    var = ds[var_name]

    # Find time dimensions
    init_time, lead_time = find_time_dimensions(ds)

    # Get coordinate values for title
    time_str = "First Ensemble Member"
    try:
        if init_time and init_time in ds.dims:
            init_val = ds[init_time].values[init_time_idx]
            time_str = f"Init: {init_val}"
        if lead_time and lead_time in ds.dims:
            lead_val = ds[lead_time].values[lead_time_idx]
            time_str += f", Lead: {lead_val}"
    except Exception as e:
        print(f"  Warning: Could not extract time values: {e}")

    # Build selection dict
    selection = {real_dim: 0}
    if init_time and init_time in var.dims:
        selection[init_time] = init_time_idx
    if lead_time and lead_time in var.dims:
        selection[lead_time] = lead_time_idx

    # Select first ensemble member
    first_member = var.isel(selection)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Determine plot type based on data dimensions
    ndim = first_member.ndim
    spatial_dims = find_spatial_dimensions(first_member)

    print(f"  Data shape: {first_member.shape}, Dimensions: {first_member.dims}")
    print(f"  Spatial dimensions found: {spatial_dims}")

    if ndim == 0:
        # Scalar value
        ax.text(
            0.5,
            0.5,
            f"Value: {first_member.values:.4f}",
            ha="center",
            va="center",
            fontsize=16,
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        im = None
    elif ndim == 1:
        # 1D data - plot as line
        first_member.plot(ax=ax, linewidth=2, color="steelblue", marker="o")
        ax.grid(True, alpha=0.3)
        ax.set_ylabel(var_name)
        im = None
    elif len(spatial_dims) >= 2:
        # Has at least 2 spatial dimensions - plot 2D
        # Select first indices of non-spatial dimensions
        for dim in first_member.dims:
            if dim not in spatial_dims[:2]:
                first_member = first_member.isel({dim: 0})

        im = first_member.plot(ax=ax, cmap="viridis", add_colorbar=False)
        units = var.attrs.get("units", "N/A")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f"{var_name} ({units})")
    elif len(spatial_dims) == 1:
        # Only 1 spatial dimension - plot as line
        first_member.plot(ax=ax, linewidth=2, color="steelblue", marker="o")
        ax.grid(True, alpha=0.3)
        im = None
    else:
        # No spatial dimensions found, try to squeeze and plot
        first_member_squeezed = first_member.squeeze()
        if first_member_squeezed.ndim == 0:
            ax.text(
                0.5,
                0.5,
                f"Value: {first_member_squeezed.values:.4f}",
                ha="center",
                va="center",
                fontsize=16,
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            im = None
        elif first_member_squeezed.ndim == 1:
            first_member_squeezed.plot(
                ax=ax, linewidth=2, color="steelblue", marker="o"
            )
            ax.grid(True, alpha=0.3)
            im = None
        else:
            im = first_member_squeezed.plot(ax=ax, cmap="viridis", add_colorbar=False)
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(var_name)

    # Add title
    ax.set_title(f"{var_name} - {time_str}")

    print("  Plot created")
    print(f"  Min value: {first_member.min().values:.4f}")
    print(f"  Max value: {first_member.max().values:.4f}")
    print(f"  Mean value: {first_member.mean().values:.4f}")

    return fig, ax


def plot_all_ensemble_members(
    ds: xr.Dataset,
    var_name: str,
    real_dim: str,
    init_time_idx: int = 0,
    lead_time_idx: int = 0,
    max_cols: int = 3,
) -> tuple:
    """
    Plot all ensemble members in a grid.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset
    var_name : str
        Variable name to plot
    real_dim : str
        Name of realization dimension
    init_time_idx : int
        Initialization time index
    lead_time_idx : int
        Lead time index
    max_cols : int
        Maximum columns in subplot grid

    Returns
    -------
    tuple
        (fig, axes) matplotlib objects
    """
    print(f"\nPlotting all ensemble members of '{var_name}'...")

    var = ds[var_name]
    n_members = ds.sizes[real_dim]

    # Find time dimensions
    init_time, lead_time = find_time_dimensions(ds)

    # Create grid
    n_cols = min(max_cols, n_members)
    n_rows = int(np.ceil(n_members / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)

    axes_flat = axes.flatten()

    # Plot each member
    vmin = var.min().values
    vmax = var.max().values

    for member_idx in range(n_members):
        # Build selection dict
        selection = {real_dim: member_idx}
        if init_time and init_time in var.dims:
            selection[init_time] = init_time_idx
        if lead_time and lead_time in var.dims:
            selection[lead_time] = lead_time_idx

        member = var.isel(selection)

        # Remove non-spatial dimensions except the first 2
        spatial_dims = find_spatial_dimensions(member)
        for dim in member.dims:
            if dim not in spatial_dims[:2]:
                member = member.isel({dim: 0})

        im = member.plot(
            ax=axes_flat[member_idx],
            cmap="viridis",
            add_colorbar=False,
            vmin=vmin,
            vmax=vmax,
        )
        axes_flat[member_idx].set_title(f"Member {member_idx}")

    # Remove unused subplots
    for idx in range(n_members, len(axes_flat)):
        fig.delaxes(axes_flat[idx])

    fig.suptitle(f"{var_name} - All {n_members} Ensemble Members", fontsize=14)
    fig.colorbar(im, ax=axes_flat[:n_members], shrink=0.8, label=var_name)
    plt.tight_layout()

    print(f"  Plot created with {n_members} members")

    return fig, axes


def plot_ensemble_statistics(
    ds: xr.Dataset,
    var_name: str,
    real_dim: str,
    init_time_idx: int = 0,
    lead_time_idx: int = 0,
) -> tuple:
    """
    Plot first member, ensemble mean, and ensemble std deviation.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset
    var_name : str
        Variable name
    real_dim : str
        Name of realization dimension
    init_time_idx : int
        Initialization time index
    lead_time_idx : int
        Lead time index

    Returns
    -------
    tuple
        (fig, axes) matplotlib objects
    """
    print(f"\nPlotting ensemble statistics for '{var_name}'...")

    var = ds[var_name]

    # Find time dimensions
    init_time, lead_time = find_time_dimensions(ds)

    # Build selection dict for time indices
    time_selection = {}
    if init_time and init_time in var.dims:
        time_selection[init_time] = init_time_idx
    if lead_time and lead_time in var.dims:
        time_selection[lead_time] = lead_time_idx

    # Select times
    var_subset = var.isel(time_selection) if time_selection else var

    # Compute statistics
    ensemble_mean = var_subset.mean(dim=real_dim)
    ensemble_std = var_subset.std(dim=real_dim)
    first_member = var_subset.isel({real_dim: 0})

    # Remove non-spatial dimensions
    spatial_dims = find_spatial_dimensions(first_member)
    for dim in first_member.dims:
        if dim not in spatial_dims[:2]:
            first_member = first_member.isel({dim: 0})
            ensemble_mean = ensemble_mean.isel({dim: 0})
            ensemble_std = ensemble_std.isel({dim: 0})

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot first member
    im0 = first_member.plot(ax=axes[0], cmap="viridis", add_colorbar=False)
    axes[0].set_title("First Ensemble Member")
    plt.colorbar(im0, ax=axes[0])

    # Plot ensemble mean
    im1 = ensemble_mean.plot(ax=axes[1], cmap="viridis", add_colorbar=False)
    axes[1].set_title("Ensemble Mean")
    plt.colorbar(im1, ax=axes[1])

    # Plot ensemble std (spread)
    im2 = ensemble_std.plot(ax=axes[2], cmap="YlOrRd", add_colorbar=False)
    axes[2].set_title("Ensemble Std Dev (Spread)")
    plt.colorbar(im2, ax=axes[2])

    fig.suptitle(f"{var_name} - Ensemble Statistics", fontsize=14)
    plt.tight_layout()

    # Print statistics
    print(f"\n  {var_name} Statistics:")
    print(
        f"    First member   - min: {first_member.min().values:.4f}, "
        f"max: {first_member.max().values:.4f}, "
        f"mean: {first_member.mean().values:.4f}"
    )
    print(
        f"    Ensemble mean  - min: {ensemble_mean.min().values:.4f}, "
        f"max: {ensemble_mean.max().values:.4f}, "
        f"mean: {ensemble_mean.mean().values:.4f}"
    )
    print(
        f"    Ensemble std   - min: {ensemble_std.min().values:.4f}, "
        f"max: {ensemble_std.max().values:.4f}, "
        f"mean: {ensemble_std.mean().values:.4f}"
    )

    return fig, axes


def process_dataset(
    data_path: str, output_dir: Optional[str] = None, variables: Optional[list] = None
) -> None:
    """
    Process a dataset: print realization info and create plots.

    Parameters
    ----------
    data_path : str
        Path to the netCDF file
    output_dir : str, optional
        Directory to save plots. If None, plots are shown but not saved
    variables : list, optional
        List of variables to plot. If None, uses all variables with realization dim
    """
    print("\n" + "=" * 70)
    print("ENSEMBLE DATA PROCESSING")
    print("=" * 70)

    # Load dataset
    print(f"\nLoading dataset: {data_path}")
    ds = xr.open_dataset(data_path)
    print("Dataset loaded")
    print(f"  Dimensions: {dict(ds.sizes)}")
    print(f"  Variables: {list(ds.data_vars.keys())}")

    # Create output directory if needed
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Output directory: {output_dir}")

    # Print realization information
    info = print_realization_info(ds, verbose=True)

    if not info["exists"]:
        print("No realization dimension found. Exiting.")
        return

    real_dim = info["dim_name"]

    # Determine which variables to plot
    if variables is None:
        variables = info["variables"]
    else:
        variables = [v for v in variables if v in info["variables"]]

    if not variables:
        print("No variables with realization dimension found.")
        return

    print(f"\nProcessing {len(variables)} variable(s)...")

    # Process each variable
    for var_name in variables:
        print(f"\n{var_name}:")

        try:
            # Plot first member
            fig1, _ = plot_first_ensemble_member(ds, var_name, real_dim)
            if output_dir:
                fig1.savefig(output_dir / f"{var_name}_first_member.png", dpi=150)
                print(f"  Saved: {var_name}_first_member.png")
            else:
                plt.show()

            # Plot all members
            fig2, _ = plot_all_ensemble_members(ds, var_name, real_dim)
            if output_dir:
                fig2.savefig(output_dir / f"{var_name}_all_members.png", dpi=150)
                print(f"  Saved: {var_name}_all_members.png")
            else:
                plt.show()

            # Plot ensemble statistics
            fig3, _ = plot_ensemble_statistics(ds, var_name, real_dim)
            if output_dir:
                fig3.savefig(output_dir / f"{var_name}_ensemble_stats.png", dpi=150)
                print(f"  Saved: {var_name}_ensemble_stats.png")
            else:
                plt.show()

            plt.close("all")
        except Exception as e:
            print(f"  Error processing {var_name}: {e}")
            import traceback

            traceback.print_exc()
            plt.close("all")
            continue

    # Summary
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    if output_dir:
        print(f"All plots saved to: {output_dir}")
    print()


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        # Default paths for cccma_ppp
        data_path = "/fs/site7/eccc/crd/cccma/users/rna002/output/prediction_integration_test_results/_trained_model/inference/predictions_2020_nstds1.0.nc"
        output_dir = "/fs/site7/eccc/crd/cccma/users/rna002/output/ensemble_plots"

    process_dataset(data_path, output_dir)
