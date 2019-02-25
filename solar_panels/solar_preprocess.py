"""
Script to preprocess the dataset:
* Creates comma-separated CSV file
* Converts images to GGB (grayscale)
* Creates subsets for training and validation
* Adds columns to indicate training or validation (useful for analysis of the deployed model)
* Creates zip archive
"""

import pandas as pd
import numpy as np
import glob
import cv2
import os
import shutil
from fire import Fire
from keras.applications import vgg19
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split

def remove_old_output(index_target_file_path, image_target_dir):
    try:
        os.remove(index_target_file_path)
        image_files = glob.glob(os.path.join(image_target_dir,'*'))
        for f in image_files:
            os.remove(f)
    except:
        pass
    try:
        if os.path.isdir(image_target_dir)==False:
            pass
            os.makedirs(image_target_dir)
    except:
        print('Cannot create target directory: %s' % (image_target_dir))
        exit(1)


def get_csv(index_source_file_path):
    return pd.read_csv(index_source_file_path,
                       delim_whitespace=True,
                       header = None,
                       names=["image", "prob", "type"])

def convert_to_rgb(df, root_dir, image_target_dir):
    for index, row in df.iterrows():
        image_source_path = os.path.join(root_dir, row['image'])
        rgb_image = cv2.imread(image_source_path)
        image_target_path = os.path.join(image_target_dir, row['image'].split('/')[1])
        assert(cv2.imwrite(image_target_path, rgb_image) == True)
    print('Converted images to RGB...')

def convert_to_npy(df, root_dir, image_target_dir):
    for index, row in df.iterrows():
        image_source_path = os.path.join(root_dir, row['image'])
        rgb_image = cv2.imread(image_source_path)
        np_data = np.array(rgb_image, dtype='f4', order='C')
        np_data_target_path = os.path.join(image_target_dir, row['image'].split('/')[1])
        np_data_target_path = np_data_target_path.replace('.png', '')
        np.save(np_data_target_path + '.npy', np_data)
    print('Converted images to NumPy...')

def convert_to_npy_vgg19(df, root_dir, image_target_dir):
    for index, row in df.iterrows():
        image_source_path = os.path.join(root_dir, row['image'])
        image = load_img(image_source_path, target_size=(300, 300))
        numpy_image = img_to_array(image)
        image_batch = np.expand_dims(numpy_image, axis=0)
        np_data = vgg19.preprocess_input(image_batch.copy())
        np_data_target_path = os.path.join(image_target_dir, row['image'].split('/')[1])
        np_data_target_path = np_data_target_path.replace('.png', '')
        np.save(np_data_target_path + '.npy', np_data[0,:,:,:])
    print('Converted images to VGG19 normalized NumPy...')

def save_as_csv(df, index_target_file_path, delimiter=','):
    df.to_csv(index_target_file_path, sep=delimiter, encoding='utf-8', index=False)

def create_subsets(df, stratify_on_type):
    if stratify_on_type == True:
        df["strata"] = df["prob"].map(str) + df["type"]
        train_data, validate_data = train_test_split(df, test_size=0.25, random_state=42, stratify=df[['strata']])
        train_data = train_data.drop(['strata'], axis=1)
        validate_data = validate_data.drop(['strata'], axis=1)
    else:
        train_data, validate_data = train_test_split(df, test_size=0.25, random_state=42, stratify=df[['prob']])
    print('Training samples: ' + str(len(train_data.values)))
    print('Validation samples: ' + str(len(validate_data.values)))
    train_data.insert(loc=3, column='subset', value='T')
    validate_data.insert(loc=3, column='subset', value='V')
    return train_data.append(validate_data, ignore_index=True)

def balance_classes(df):
    df_sample_training = df[(df['subset'] == 'T') & (df['prob'] == 1.0)]
    df_sample_validation = df[(df['subset'] == 'V') & (df['prob'] == 1.0)]
    print('Upsampled defects...')
    return pd.concat([df, df_sample_training, df_sample_validation], axis=0, sort=False)

def add_binary_label(df):
    # Add column for use in classification models
    df['prob_binary'] = df['prob'].apply(lambda x: 0 if x==0.0 else 1)
    print('Added binary label...')
    return df

def add_rotated_samples(df, root_dir, image_target_dir):
    for index, row in df.iterrows():
        image_source_path = os.path.join(root_dir, row['image'])
        image = cv2.imread(image_source_path)
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        rotate_filename = 'rot_' + row['image'].split('/')[1]
        image_target_path = os.path.join(image_target_dir, rotate_filename)
        assert (cv2.imwrite(image_target_path, rotated_image) == True)
    df2 = df.copy()
    df2['image'].replace({'cell': 'rot_cell'}, inplace=True, regex=True)
    print('Added rotated duplicates...')
    return pd.concat([df, df2])

def create_zip(archive_base_dir):
    shutil.make_archive(base_name=archive_base_dir,
                        format="zip",
                        root_dir=archive_base_dir,
                        base_dir=archive_base_dir)
    print('Zip file: %s.zip' % (archive_base_dir)
)

def run(root_dir,
        rotate=False,
        stratify_on_type=False,
        image_as_np=False,
        image_as_vgg19=True,
        balance=False):
    archive_base_dir = os.path.join(root_dir, 'preprocessed')
    image_target_dir = os.path.join(root_dir, 'preprocessed/images')
    index_source_file_path = os.path.join(root_dir, 'labels.csv')
    index_target_file_path = os.path.join(root_dir, 'preprocessed/index.csv')

    remove_old_output(index_target_file_path, image_target_dir)
    df = get_csv(index_source_file_path)
    if image_as_vgg19 == True:
        convert_to_npy_vgg19(df, root_dir, image_target_dir)
        df['image'] = df['image'].str.replace('.png', '.npy')
    elif image_as_np == True:
        convert_to_npy(df, root_dir, image_target_dir)
        df['image'] = df['image'].str.replace('.png', '.npy')
    else:
        convert_to_rgb(df, root_dir, image_target_dir)
    df = create_subsets(df, stratify_on_type)
    if rotate == True:
        df = add_rotated_samples(df, root_dir, image_target_dir)
    if balance == True:
        df = balance_classes(df)
    df = add_binary_label(df)
    print('Total samples: ' + str(len(df.values)))
    save_as_csv(df, index_target_file_path)
    create_zip(archive_base_dir)
    print('Done.')

if __name__ == "__main__":
    Fire(run)