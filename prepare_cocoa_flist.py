import argparse
import os
from icecream import ic
import json

parser = argparse.ArgumentParser()
parser.add_argument('--folder_path', default='./dataset/COCOA/', type=str,
                    help='The folder path')
parser.add_argument('--train_filename', default='./dataset/COCOA/train.txt', type=str,
                    help='The output filename.')
parser.add_argument('--val_filename', default='./dataset/COCOA/val.txt', type=str,
                    help='The output filename.')

if __name__ == "__main__":

    args = parser.parse_args()

    # make 2 lists to save file paths
    train_file_names = []
    val_file_names = []

    # Opening JSON file
    train_f = open(args.folder_path + 'annotations/COCO_amodal_train2014.json')
    train_data = json.load(train_f)
    for img in train_data['images']:
        f_name = 'train2014/' + img['file_name']
        train_file_names.append(f_name)
    train_f.close()

    # Opening JSON file
    val_f = open(args.folder_path + 'annotations/COCO_amodal_val2014.json')
    val_data = json.load(val_f)
    for img in val_data['images']:
        f_name = 'val2014/' + img['file_name']
        val_file_names.append(f_name)
    val_f.close()
        
    ic(len(train_file_names))
    ic(len(val_file_names))

    if not os.path.exists(args.train_filename):
        os.mknod(args.train_filename)
    
    if not os.path.exists(args.train_filename):
        os.mknod(args.train_filename)

    fo = open(args.train_filename, "w")
    fo.write("\n".join(train_file_names))
    fo.close()
    
    fo = open(args.val_filename, "w")
    fo.write("\n".join(val_file_names))
    fo.close()