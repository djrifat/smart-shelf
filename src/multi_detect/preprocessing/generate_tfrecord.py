"""
Generate tensorflow record from csv file
Usage:
    # Create train data:
    python generate_tfrecord.py --csv_input=../annotations/train_labels.csv --output_path=../annotations/train.record --image_dir=../images/train/

    # Create test data:
    python generate_tfrecord.py --csv_input=../annotations/test_labels.csv --output_path=../annotations/test.record --image_dir=../images/test/ 
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict
import tensorflow as tf
from PIL import Image
import pandas as pd
import os
import io

#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
#print(tf.compat.v1.__version__)

# Set tensorlow flags
flags = tf.compat.v1.flags
flags.DEFINE_string('csv_input','','Path to CSV input location')
flags.DEFINE_string('output_path','','Path to output TFRecord location')
flags.DEFINE_string('image_dir','','Path to images')
FLAGS = flags.FLAGS

# Convert class labels to integer values 
# @param row_label {str}: Class row value
# @return {int}: Return integer value of each corresponding class label  
def class_text_to_int(row_label):
    if row_label == 'hand':
        return 1
    elif row_label == "facew":
        return 2
    else:
        return None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def create_tf_example(group, path):
    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    x_mins = []
    x_maxs = []
    y_mins = []
    y_maxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        x_mins.append(row['xmin'] / width)
        x_maxs.append(row['xmax'] / width)
        y_mins.append(row['ymin'] / height)
        y_maxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(x_mins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(x_maxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(y_mins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(y_maxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(FLAGS.image_dir)
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.compat.v1.app.run()
