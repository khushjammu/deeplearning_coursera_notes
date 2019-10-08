import tensorflow as tf

dataset = './kc_house_data.csv'

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def read_from_csv(filename_queue):
    reader = tf.TextLineReader(skip_header_lines=1)
    _, csv_row = reader.read(filename_queue)
    record_defaults = [[0], [0], [0.], [0], [0], [0.], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0.], [0.], [0], [0]]
    price,bedrooms,bathrooms,sqft_living,sqft_lot,floors,waterfront,view,condition,grade,sqft_above,sqft_basement,yr_built,yr_renovated,zipcode,lat,longi,sqft_living15,sqft_lot15 = tf.decode_csv(csv_row, record_defaults=record_defaults) 
    #colHour,colQuarter,colAction,colUser,colLabel = tf.decode_csv(csv_row, record_defaults=record_defaults)
    features = tf.TensorArray(tf.float32, size = [0. None]).pack([bedrooms,bathrooms,sqft_living,sqft_lot,floors,waterfront,view,condition,grade,sqft_above,sqft_basement,yr_built,yr_renovated,zipcode,lat,longi,sqft_living15,sqft_lot15])  
    label = tf.TensorArray(tf.int32).pack([price])  
    return features, label

def input_pipeline(batch_size, num_epochs=None):
    filename_queue = tf.train.string_input_producer([dataset], num_epochs=num_epochs, shuffle=True)  
    example, label = read_from_csv(filename_queue)
    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * batch_size
    example_batch, label_batch = tf.train.shuffle_batch(
        [example, label], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue)
    return example_batch, label_batch

file_length = file_len(dataset) - 1
examples, labels = input_pipeline(file_length, 1)

with tf.Session() as sess:
    init_op = tf.group(tf.initialize_all_variables(),
                   tf.initialize_local_variables())
    tf.initialize_all_variables().run()

    # start populating filename queue
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        while not coord.should_stop():
            example_batch, label_batch = sess.run([examples, labels])
        print(example_batch)
    except tf.errors.OutOfRangeError:
        print('Done training, epoch reached')
    finally:
        coord.request_stop()

    coord.join(threads) 

