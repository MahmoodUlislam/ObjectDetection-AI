import argparse
from src import train, evaluate, inference


def main():
    parser = argparse.ArgumentParser(description="Object Detection Project")
    parser.add_argument('--mode', type=str, required=True, help='train, evaluate, or inference')
    args = parser.parse_args()

    if args.mode == 'train':
        train.run()
    elif args.mode == 'evaluate':
        evaluate.run()
    elif args.mode == 'inference':
        inference.run()
    else:
        print("Invalid mode. Choose from 'train', 'evaluate', or 'inference'.")


if __name__ == '__main__':
    main()
