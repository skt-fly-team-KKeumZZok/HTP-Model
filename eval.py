import tensorflow as tf
import matplotlib.pyplot as plt
from func import my_func

# Create a TFRecordDataset object from your TFRecord file
tfrecord_path = 'HTP_data/test/house.tfrecord'
dataset = tf.data.TFRecordDataset(tfrecord_path)

def IoU(box1, box2):
    # box = (x1, y1, x2, y2)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # obtain x1, y1, x2, y2 of the intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # compute the width and height of the intersection
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)

    inter = w * h
    iou = inter / (box1_area + box2_area - inter)
    return iou

def cordinate_iou():
    import matplotlib
    import matplotlib.pyplot as plt

    import io
    import os
    import gc; gc.collect()
    import pathlib
    import scipy.misc
    import numpy as np
    from six import BytesIO
    from PIL import Image, ImageDraw, ImageFont

    import tensorflow as tf

    from object_detection.utils import label_map_util
    from object_detection.utils import config_util
    from object_detection.utils import visualization_utils as viz_utils
    from object_detection.builders import model_builder

    def load_image_into_numpy_array(img):
        # img_data = tf.io.gfile.GFile(path, 'rb').read()
        image = Image.open(BytesIO(img))
        (im_width, im_height) = image.size
        
        return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

    #recover our saved model
    pipeline_config = 'fine_tuned_model/house_pipeline.config'

    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    model_config = configs['model']
    detection_model = model_builder.build(
        model_config=model_config, is_training=False)

    # # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(
            model=detection_model)

    ckpt.restore('training/ckpt-H').expect_partial()

    def get_model_detection_function(model):
        """Get a tf.function for detection."""

        @tf.function
        def detect_fn(image):
            """Detect objects in image."""

            image, shapes = model.preprocess(image)
            prediction_dict = model.predict(image, shapes)
            detections = model.postprocess(prediction_dict, shapes)

            return detections, prediction_dict, tf.reshape(shapes, [-1])

        return detect_fn

    detect_fn = get_model_detection_function(detection_model)

    #map labels for inference decoding
    label_map_path = configs['eval_input_config'].label_map_path
    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(
        label_map,
        max_num_classes=label_map_util.get_max_label_map_index(label_map),
        use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)

    # image load
    xy_list = []
    for raw_record in dataset:
 
        example = tf.io.parse_single_example(raw_record, feature_description)

        decoded_image = tf.image.decode_jpeg(example["image/encoded"], channels=3)
        jpeg_image = tf.image.encode_jpeg(decoded_image)
        img = jpeg_image.numpy()

        image_np = load_image_into_numpy_array(img)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections, predictions_dict, shapes = detect_fn(input_tensor)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'][0].numpy(),
                (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
                detections['detection_scores'][0].numpy(),
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=.5,
                agnostic_mode=False
                )
        
        score = detections['detection_scores'][0].numpy()
        top_score = score[score > 0.5]
        box_cnt = len(top_score)
        top_classes = detections['detection_classes'][0][:box_cnt].numpy()
        top_boxes = detections['detection_boxes'][0][:box_cnt].numpy()

        xy = []
        for i in range(box_cnt):
            top_boxes = detections['detection_boxes'][0][i].numpy()

            x1_point = top_boxes[0]
            y1_point = top_boxes[1]
            x2_point = top_boxes[2]
            y2_point = top_boxes[3]
            
            xy.append([y1_point, x1_point, y2_point, x2_point])

        xy_list.append(xy)

        print(top_classes)
        print(xy_list)

    # return xy_list



#-------------------------------------------------------#

# 정답 좌표
# Define the feature description for your dataset
feature_description = {
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/filename': tf.io.FixedLenFeature([], tf.string),
    'image/height': tf.io.FixedLenFeature([], tf.int64),
    'image/width': tf.io.FixedLenFeature([], tf.int64),
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    'image/object/class/text': tf.io.VarLenFeature(tf.string),
    'image/object/class/label': tf.io.VarLenFeature(tf.int64)}


# Parse a single example from the dataset
iou_list = []
k = 0

import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.style.use('default')
plt.rcParams['figure.figsize'] = (4, 3)
plt.rcParams['font.size'] = 12

for row in dataset:

    example = tf.io.parse_single_example(row, feature_description)

    # Extract the bounding box coordinates
    xmin = tf.sparse.to_dense(example['image/object/bbox/xmin']).numpy()
    ymin = tf.sparse.to_dense(example['image/object/bbox/ymin']).numpy()
    xmax = tf.sparse.to_dense(example['image/object/bbox/xmax']).numpy()
    ymax = tf.sparse.to_dense(example['image/object/bbox/ymax']).numpy()

    label = tf.sparse.to_dense(example['image/object/class/label']).numpy()
    print(label)

    # Print the coordinates
    true_box = []
    # plt.figure()
    # currentAxis = plt.gca()
    for i in range(len(xmin)):
        true_box.append([xmin[i], ymin[i], xmax[i], ymax[i]])
        # currentAxis.add_patch(
        # patches.Rectangle(
        #     (xmin[i], ymin[i]),                                   # (x, y)
        #     xmax[i]-xmin[i], ymax[i]-ymin[i],                     # width, height
        #     edgecolor = 'deeppink',
        #     facecolor = 'lightgray',
        #     fill=False
        # ))

    # plt.show()
    print(true_box)
    # pred_box = cordinate_iou()
    # print(pred_box)

    for i in range(len(true_box)):
        iou = IoU(pred_box[k][i], true_box[i])
        iou_list.append(iou)
 
    k += 1

print(iou_list)

cordinate_iou()

