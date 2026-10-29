from S2D4P.train.train import get_parser, main

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args.config)
