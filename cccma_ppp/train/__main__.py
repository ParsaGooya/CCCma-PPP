from pathlib import Path

from cccma_ppp.generic.monitoring import monitor
from cccma_ppp.train.train import get_parser, main


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    output_dir = Path("output") / "monitoring"
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config_name = Path(args.config).stem

    csv_path = output_dir / f"{config_name}_monitoring.csv"
    plot_path = output_dir / f"{config_name}_monitoring.png"

    resource_monitor = monitor(
        cpu=True,
        ram=True,
        gpu0=True,
        interval=0.1,
    )

    try:
        with resource_monitor:
            with resource_monitor.span("training"):
                main(args.config)

            resource_monitor.checkpoint("training_finished")

    finally:
        resource_monitor.stop()

        monitoring_data = resource_monitor.get_dataframe()

        if not monitoring_data.empty:
            monitoring_data.to_csv(
                csv_path,
                index=False,
            )

            resource_monitor.plot(
                df=monitoring_data,
                save_path=plot_path,
                show=False,
                smooth="kalman",
                process_variance=1.0,
                measurement_variance=25.0,
            )

            print(f"Monitoring data: {csv_path}")
            print(f"Monitoring plot: {plot_path}")
