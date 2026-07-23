from cccma_ppp.generic.monitoring import monitor
from cccma_ppp.train.train import main, get_parser

m = monitor(cpu=True, ram=True, gpus=[0, 1], interval=0.1)
m.start()

try:
    if __name__ == "__main__":
        with m.span("Running Pipeline"):
            parser = get_parser()
            args = parser.parse_args()
            main(args.config)
finally:
    m.stop()

    df = m.get_dataframe()

    m.plot(
        df,
        save_path="scripts/resource_usage.png",
        show=False,
        smooth="kalman",
    )
