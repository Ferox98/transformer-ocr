import pandas as pd
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
import os 
import numpy as np
import h5py
"""
    Folder structure:
                    dataset:
                            iam:
                                labels:

                                lines:
                            iiit:
                                groundtruth:



""" 

def prepare_iiit(data_dir='./dataset/iiit'):

    """
    Download Ground truth files and IIIT-HWS image corpus from the iiit-dataset and extract in the dataset/iiit directory
    """
    
    file_dir = os.path.join(data_dir,'groundtruth/IIIT-HWS-90K.mat')

    train_f = h5py.File(file_dir,'r')
    train_text = train_f['list/ALLtext']

    train_word_arr = []
    for i in range(train_text.shape[0]):
        vals = train_text[i][:]
        str1 = ''.join(chr(i[0]) for i in train_f[vals[0]][:])
        train_word_arr.append(str1)
    df = pd.DataFrame({'Word': train_word_arr})

    df.dropna(axis=0,inplace=True)

    df['path'] = df.apply(lambda x: str(os.path.join(os.path.join('Images_90K_Normalized' , data_dir) , str(x.index +1) )))

    
    train_df, test_df = train_test_split(df, test_size=0.1,random_state=0)
    # we reset the indices to start from zero
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    train_df.to_csv(r'./dataset/train_iiit_1.txt', header=None, index=None, sep='\t')
    test_df.to_csv(r'./dataset/test_iiit_1.txt', header=None, index=None, sep='\t')


def prepare_iam(data_dir='./dataset/iam'):

    """
    Preprocess the IAM dataset:
    Download the data/lines.tgz and data/xml.tgz files from the IAM dataset page and extract to the dataset/IAM/lines and  dataset/IAM/labels directorys respectively
    """

    label_dir = os.path.join(data_dir,'labels')
    lines_dir = os.path.join(data_dir,'lines')

    label_dirs = os.listdir(label_dir)
    results = []
    for label_file in label_dirs:
        cur_path = os.path.join(label_dir, label_file)
        if os.path.isfile(cur_path):
            # print(label_file)
            with open(cur_path, 'rb') as f:
                data = f.read()
                bs_data = BeautifulSoup(data, 'xml')
                # print([tag.name for tag in bs_data.find_all()])
                handwritten = bs_data.find_all('handwritten-part')
                if len(handwritten) == 0:
                    print(label_file)
                    continue
                handwritten = handwritten[0]
                # find all unique lines
                lines = handwritten.find_all('line')
                for line in lines:
                    line_label = line.get('text')
                    img_name = line.get('id')
                    img_path = f'{label_file[:3]}/{label_file[:-4]}/{img_name}.png'
                    full_path = str(os.path.join(lines_dir, img_path))
                    results.append([full_path, line_label])
    print(len(results))
    df = pd.DataFrame(results, columns=['file_name', 'text'])
    train_df, test_df = train_test_split(df, test_size=0.1,random_state=0)
    # we reset the indices to start from zero
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    train_df.to_csv(r'./dataset/train_iam_lines.txt', header=None, index=None, sep='\t')
    test_df.to_csv(r'./dataset/test_iam_lines.txt', header=None, index=None, sep='\t')    

# df = pd.read_fwf('../IAM/gt_test.txt', header=None)
# df = pd.read_fwf('/home/rufael.marew/Documents/Academics/ML701/project/label_words_train.csv', header=0)

# df.rename(columns={0: "file_name", 1: "text"}, inplace=True)
# del df[2]
#some file names end with jp instead of jpg, let's fix this
# df['file_name'] = df['file_name'].apply(lambda x: x + 'g' if x.endswith('jp') else x)

# train_df, test_df = train_test_split(df, test_size=0.1,random_state=0)
# # we reset the indices to start from zero
# train_df.reset_index(drop=True, inplace=True)
# test_df.reset_index(drop=True, inplace=True)

# df.to_csv(r'./dataset/data.txt', header=None, index=None, sep='\t', mode='a')
# test_df.to_csv(r'./dataset/test_1.txt', header=None, index=None, sep='\t', mode='a')
# prepare_iam('/home/rufael.marew/Documents/Academics/ML701/project/label/', '/home/rufael.marew/Documents/Academics/ML701/project/Lines/')
import argparse

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--data_dir", "-d", type=str, default="dataset/iam",
#                         help="Dataset directory")

#     parser.add_argument("--iiit", "-i", type=bool, default=False,
#                         help="wheather to preprocess iiit or iam. if set it will preprocess iiit")

#     args = parser.parse_args()

#     if args.iiit:
#         prepare_iit(args.data_dir)
#     else:
#         prepare_iam(args.data_dir)

prepare_iiit()
