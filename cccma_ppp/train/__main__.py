from cccma_ppp.train.train import get_parser, main

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args.config)
