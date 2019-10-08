from __future__ import print_function
import numpy as np
import tensorflow as tf
import math as math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dataset')
args = parser.parse_args()

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def read_from_csv(filename_queue):
  reader = tf.TextLineReader(skip_header_lines=1)
  _, csv_row = reader.read(filename_queue)
  record_defaults = [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]]
  price,bedrooms,bathrooms,sqft_living,sqft_lot,floors,waterfront,view,condition,grade,sqft_above,sqft_basement,yr_built,yr_renovated,zipcode,lat,longi,sqft_living15,sqft_lot15 = tf.decode_csv(csv_row, record_defaults=record_defaults)
  #colHour,colQuarter,colAction,colUser,colLabel = tf.decode_csv(csv_row, record_defaults=record_defaults)
  features = tf.stack([bedrooms,bathrooms,sqft_living,sqft_lot,floors,waterfront,view,condition,grade,sqft_above,sqft_basement,yr_built,yr_renovated,zipcode,lat,longi,sqft_living15,sqft_lot15])  
  label = tf.stack([price])  
  return features, label

def input_pipeline(batch_size, num_epochs=None):
  #epochs = 1
  filename_queue = tf.train.string_input_producer([args.dataset], num_epochs=num_epochs , shuffle=True)  #num_epochs, shuffle=True)  
  example, label = read_from_csv(filename_queue)
  min_after_dequeue = 10000
  capacity = min_after_dequeue + 3 * batch_size
  example_batch, label_batch = tf.train.shuffle_batch(
      [example, label], batch_size=batch_size, capacity=capacity,
      min_after_dequeue=min_after_dequeue)
  return example_batch, label_batch

file_length = file_len(args.dataset) - 1
print("Original file_length:" + str(file_length))
#file_length = 10 # import just the first 100
#print("Debugging file_length:" + str(file_length))

examples, labels = input_pipeline(file_length, 1)

with tf.Session() as sess:
  #tf.initialize_all_variables().run()
  #tf.initialize_all_variables()
  sess.run(tf.local_variables_initializer())
  sess.run(tf.global_variables_initializer())
  # start populating filename queue
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)


  try:
    while not coord.should_stop():
      example_batch, label_batch = sess.run([examples, labels])
      print("example_batch:", str(example_batch))
      print("label_batch:",str(label_batch))
  except tf.errors.OutOfRangeError:
    print('Done training, epoch reached')
  finally:
    coord.request_stop()

  coord.join(threads) 

  #print(sess.run(examples))