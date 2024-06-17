import os
import logging
from os.path import join, isfile
import numpy as np
import h5py
from glob import glob
import torchfile
from PIL import Image 
import yaml

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("Loading configuration file")
    with open('birds_dataset/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    images_path = config['birds_images_path']
    embedding_path = config['birds_embedding_path']
    text_path = config['birds_text_path']
    datasetDir = config['birds_dataset_path']

    val_classes = open(config['val_split_path']).read().splitlines()
    train_classes = open(config['train_split_path']).read().splitlines()
    test_classes = open(config['test_split_path']).read().splitlines()

    f = h5py.File(datasetDir, 'w')
    train = f.create_group('train')
    valid = f.create_group('valid')
    test = f.create_group('test')

    for _class in sorted(os.listdir(embedding_path)):
        logging.info(f"Processing class: {_class}")
        split = ''
        if _class in train_classes:
            split = train
        elif _class in val_classes:
            split = valid
        elif _class in test_classes:
            split = test

        data_path = os.path.join(embedding_path, _class)
        txt_path = os.path.join(text_path, _class)
        
        for example, txt_file in zip(sorted(glob(data_path + "/*.t7")), sorted(glob(txt_path + "/*.txt"))):
            try:
                logging.debug(f"Loading example: {example}")
                example_data = torchfile.load(example)
                
                # Ensure keys are bytes literals
                img_path = example_data[b'img']
                embeddings = example_data[b'txt']  # No .numpy() call here
                example_name = img_path.decode('utf-8').split('/')[-1][:-4]  # Convert bytes to string and remove extension

                with open(txt_file, "r") as f:
                    txt = f.readlines()

                img_path = os.path.join(images_path, img_path.decode('utf-8'))  # Convert bytes to string
                logging.debug(f"Loading image: {img_path}")
                with open(img_path, 'rb') as img_file:
                    img = img_file.read()

                txt_choice = np.random.choice(range(10), 5)

                embeddings = embeddings[txt_choice]
                txt = np.array(txt)
                txt = txt[txt_choice]
                dt = h5py.special_dtype(vlen=str)

                for c, e in enumerate(embeddings):
                    ex = split.create_group(example_name + '_' + str(c))
                    ex.create_dataset('name', data=example_name)
                    ex.create_dataset('img', data=np.void(img))
                    ex.create_dataset('embeddings', data=e)
                    ex.create_dataset('class', data=_class)
                    ex.create_dataset('txt', data=txt[c].astype(object), dtype=dt)

                logging.info(f"Processed example: {example_name}")
            except Exception as e:
                logging.error(f"Error processing example {example}: {e}")

if __name__ == "__main__":
    main()
