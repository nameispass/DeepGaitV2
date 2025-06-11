import argparse
import logging
import os
import pickle
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

def imgs2pickle_single_folder(input_path: Path, output_pkl_path: Path, img_size: int = 64, verbose: bool = False) -> None:
    """
    Read all images from a folder, preprocess them, and save into a single pkl file.
    """
    to_pickle = []
    logging.info(f'Scanning images in {input_path}...')
    img_files = sorted(list(input_path.rglob('*.jpg')))
    logging.info(f'Found {len(img_files)} images.')

    for img_file in tqdm(img_files, desc='Processing images'):
        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)

        if img is None:
            logging.warning(f"Cannot read {img_file}. Skipping.")
            continue

        if img.sum() <= 10000:
            logging.warning(f'{img_file} has no data.')
            continue

        y_sum = img.sum(axis=1)
        y_top = (y_sum != 0).argmax()
        y_btm = (y_sum != 0).cumsum().argmax()
        img = img[y_top:y_btm + 1, :]

        ratio = img.shape[1] / img.shape[0]
        img = cv2.resize(img, (int(img_size * ratio), img_size), interpolation=cv2.INTER_CUBIC)

        x_csum = img.sum(axis=0).cumsum()
        x_center = None
        for idx, csum in enumerate(x_csum):
            if csum > img.sum() / 2:
                x_center = idx
                break

        if x_center is None:
            logging.warning(f'{img_file} has no center.')
            continue

        half_width = img_size // 2
        left = x_center - half_width
        right = x_center + half_width
        if left <= 0 or right >= img.shape[1]:
            pad = np.zeros((img.shape[0], half_width), dtype=img.dtype)
            img = np.concatenate([pad, img, pad], axis=1)
            left += half_width
            right += half_width

        cropped = img[:, left:right].astype('uint8')
        to_pickle.append(cropped)

    if len(to_pickle) == 0:
        logging.warning("No valid images to save.")
        return

    to_pickle = np.array(to_pickle)
    os.makedirs(output_pkl_path.parent, exist_ok=True)  # <--- thêm dòng này

    with open(output_pkl_path, 'wb') as f:
        pickle.dump(to_pickle, f)
    logging.info(f'Saved {len(to_pickle)} images to {output_pkl_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple pretreatment for one folder into 090.pkl')
    parser.add_argument('-i', '--input_path', type=str, required=True, help='Input folder containing images.')
    parser.add_argument('-o', '--output_path', type=str, required=True, help='Output path to save 090.pkl.')
    parser.add_argument('-r', '--img_size', default=64, type=int, help='Image resize height. Default: 64')
    parser.add_argument('-v', '--verbose', default=False, action='store_true', help='Verbose output.')

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format='[%(asctime)s - %(levelname)s]: %(message)s')

    input_path = Path(args.input_path)
    output_pkl_path = Path(args.output_path) / '090.pkl'

    imgs2pickle_single_folder(input_path, output_pkl_path, img_size=args.img_size, verbose=args.verbose)
