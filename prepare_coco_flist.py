import argparse
import os
from icecream import ic

parser = argparse.ArgumentParser()
parser.add_argument('--folder_path', default='./dataset/COCOA', type=str,
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

    dir_item_train = os.listdir(os.path.join(args.folder_path, 'train2014'))
    for di in dir_item_train: 
        f_name = 'train2014' + "/" + di
        train_file_names.append(f_name)
    
    dir_item_val = os.listdir(os.path.join(args.folder_path, 'val2014'))
    for di in dir_item_val: 
        f_name = 'val2014' + "/" + di
        val_file_names.append(f_name)
    
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