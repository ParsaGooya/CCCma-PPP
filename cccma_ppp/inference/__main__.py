from cccma_ppp.inference.predict import main, get_parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args.config)