import argparse

from src.train import run as train_run
from src.evaluate import run as evaluate_run
from src.inference import run as inference_run


def main():
    parser = argparse.ArgumentParser(description="Object Detection Project")
    parser.add_argument('--mode', type=str, required=True, help='train, evaluate, or inference')
    args = parser.parse_args()

    if args.mode == 'train':
        train_run()
    elif args.mode == 'evaluate':
        evaluate_run()
    elif args.mode == 'inference':
        inference_run('assets/apple.jpg')
    else:
        print("Invalid mode. Choose from 'train', 'evaluate', or 'inference'.")


if __name__ == '__main__':
    main()
