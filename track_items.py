"""Uses detection and classification models to generate labeled bounding boxes in csv format.
"""

import os
import glob
from typing import Any

from absl import app
from absl import flags
from cvx2 import latest as cv2
import numpy as np
import pandas as pd
import PIL.Image as Image
from skimage import util
import tensorflow as tf

import video_util


FLAGS = flags.FLAGS

flags.DEFINE_string('input_video', '', 'The path of the video.')
flags.DEFINE_string('output_csv', '',
                    'Path for output csv file.')
flags.DEFINE_string('output_file_dir', '',
                    'Path for output file.')
flags.DEFINE_float('fps', 30,
                   'The number of frames per second to be processed.')


_PATH_TO_DETECTION_MODEL = '.../XX.tflite'
_PATH_TO_CLASSIFICATION_MODEL = '.../XX.tflite'
_ITEMS_CLASS_TYPE = []


BoxNorm = tuple[float, float, float, float]
Box = tuple[int, int, int, int]


def write_csv_file(path: str, df: pd.DataFrame, header: bool = True) -> None:
  folder = os.path.dirname(path)
  if not os.path.exists(folder):
    os.makedirs(folder_path, exist_ok=True)
  with open(path, 'w') as fh:
    df.to_csv(fh, index=False, header=header)


def save_frames(frames: list[np.ndarray], image_base_path: str,
                file_id: str) -> None:
  """Saves frames on the provided image_base_path with the name file_id."""
  for count, frame in enumerate(frames):
    img_pil = Image.fromarray(np.uint8(frame))
    img_save_path = (
        f'{image_base_path}/{file_id}/{file_id.zfill(4)}_{str(count).zfill(6)}.png'
    )
    existing_img_paths = glob.glob(
        os.path.join(f'{image_base_path}/{file_id}', '*'), recursive=True
    )
    if img_save_path not in existing_img_paths:
      with open(img_save_path, 'wb') as f:
        img_pil.save(f, 'PNG')


def filter_overlapping_low_confidence_boxes(
    bboxs: list[Box],
    confidences: list[float],
    items_np_list: list[np.ndarray],
    frame_id: int,
    overlap_threshold: float = 0.4,
) -> list[tuple[int, np.ndarray, int, int, int, int]]:
  """Remove overlap box which has lower confidence.

  Args:
    bboxs: bboxs list by pixel
    confidences: boxes confidence list
    items_np_list: original items np list
    frame_id: current frame
    overlap_threshold: overlap thresholds

  Returns:
    Items with no overlap boxes.
  """
  boxes = np.array(bboxs)
  x1 = boxes[:, 1]  # x coordinate of the top-left corner
  y1 = boxes[:, 0]  # y coordinate of the top-left corner
  x2 = boxes[:, 3]  # x coordinate of the bottom-right corner
  y2 = boxes[:, 2]  # y coordinate of the bottom-right corner
  areas = (x2 - x1 + 1) * (y2 - y1 + 1)
  indices = np.arange(len(x1))
  items = []
  for i, box in enumerate(boxes):
    temp_indices = [x for x in indices if x != i]
    # Find out the coordinates of the intersection box
    xx1 = np.maximum(box[1], boxes[temp_indices, 1])
    yy1 = np.maximum(box[0], boxes[temp_indices, 0])
    xx2 = np.minimum(box[3], boxes[temp_indices, 3])
    yy2 = np.minimum(box[2], boxes[temp_indices, 2])
    # Find out the width and the height of the intersection box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)
    # compute the ratio of overlap
    overlap = (w * h) / areas[temp_indices]
    confidence = confidences[i]
    if np.any(overlap > overlap_threshold):
      detail_overlap = np.asarray(overlap > overlap_threshold).nonzero()
      remove_current = False
      for _, overlap_box_index in enumerate(detail_overlap[0]):
        temp_index = temp_indices[int(overlap_box_index)]
        overlap_confidence = confidences[temp_index]
        if (overlap_confidence > confidence) or (
            (overlap_confidence == confidence)
            and (areas[temp_index] < areas[i])
        ):
          remove_current = True
        else:
          indices = indices[indices != temp_index]
      if remove_current:
        indices = indices[indices != i]
  for i in indices.tolist():
    if items_np_list[i].shape[0] > 0 and items_np_list[i].shape[1] > 0:
      items.append((
          frame_id,
          items_np_list[i],
          boxes[i][1],
          boxes[i][0],
          boxes[i][3],
          boxes[i][2],
      ))
  return items


def detect_items(
    frames: list[np.ndarray],
) -> list[tuple[int, np.ndarray, int, int, int, int]]:
  """Object detection."""
  tf_model = video_util.DetectionTFModel(_PATH_TO_DETECTION_MODEL)
  objects = []
  for frame_id, frame in enumerate(frames):
    output = tf_model.infer(frame)
    boxes_norm, names, confidences = zip(*output)
    object_id = 1
    boxes_norm_filtered = [
        box for name, box in zip(names, boxes_norm) if name == object_id
    ]
    boxes = [
        video_util.box_norm_to_box(box_norm, frame)
        for box_norm in boxes_norm_filtered
    ]
    cropped_images = [
        video_util.box_to_cropped_image(box, frame) for box in boxes
    ]
    filtered_objects = filter_overlapping_low_confidence_boxes(
        boxes, confidences, cropped_images, frame_id
    )
    objects.extend(filtered_objects)
  return objects


def classify_defect(
    image_patch: np.ndarray, interpreter: tf.lite.Interpreter
) -> float:
  """Object classification."""
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  shape_list = input_details[0].get('shape').tolist()
  image = resize_np_image(image_patch, shape_list, as_ubyte=True)
  interpreter.set_tensor(input_details[0]['index'], image)
  interpreter.invoke()
  score_list = interpreter.get_tensor(output_details[0]['index']).tolist()
  finals = []
  for scores in score_list:
    total_score = sum(scores)
    score_percent = [float(item) / total_score for item in scores]
    score_labels = list(zip(score_percent, _ITEMS_CLASS_TYPE))
    score_labels.sort(reverse=True)
    score, label = score_labels[0]
    finals.append((label, score))
  return finals[0][0]


def load_tflite_model(model_path: str) -> tf.lite.Interpreter:
  with open(model_path, 'rb') as fh:
    content = fh.read()
  interpreter = tf.lite.Interpreter(model_content=content)
  interpreter.allocate_tensors()
  return interpreter


def resize_np_image(
    image: np.ndarray, shape_list: list[int], as_ubyte: bool = False
) -> np.ndarray:
  image = cv2.resize(
      image, tuple(shape_list[1:3]), interpolation=cv2.INTER_AREA
  )
  if image.shape[2] != cv2.INTER_AREA:
    image = image[:, :, 0:3]

  image = image.reshape(shape_list)
  if as_ubyte:
    image = util.img_as_ubyte(image)
  return image


def iou_calculate(
    box1: Box, boxes_and_other_stuff: list[list[Any]],
    iou_threshold: float = 0.2,
) -> tuple[bool, int, list[Any]]:
  """Compare IOU of current box with previous boxes."""
  update_df = boxes_and_other_stuff[:]
  id_list = [value[-1] for value in boxes_and_other_stuff]

  boxes = np.array([bbox[1:5] for bbox in boxes_and_other_stuff])
  ious = [video_util.compute_iou(box1, box) for box in boxes]
  max_box_id = np.argmax(ious)

  above_threshold = max(ious) > iou_threshold
  is_new = False if above_threshold else True
  object_id = int(id_list[max_box_id].split('_')[-1]) if above_threshold else -1
  if above_threshold:
    update_df.pop(max_box_id)
  return is_new, object_id, update_df


def run_track(input_video: str, output_path: str, output_file_dir: str) -> None:
  """Track objects in video."""
  output_path = f'{output_path}.csv'
  file_id = output_path.split('/')[-1].split('.')[0]
  image_base_path = f'{output_file_dir}/image_data'

  frames = video_util.video_to_frames(input_video, FLAGS.fps)
  save_frames(frames, image_base_path, file_id)
  print('[+] Saved images done.')
  results = detect_items(frames)
  print('[+] Detection done.')
  classify_interpreter = load_tflite_model(_PATH_TO_CLASSIFICATION_MODEL)
  classify_result = []
  for frame_id, patch, x1, y1, x2, y2 in results:
    classify_result.append(
        (frame_id, x1, y1, x2, y2, classify_defect(patch, classify_interpreter))
    )
  print('[+] Classification done.')
  result_df = pd.DataFrame(
      classify_result,
      columns=['frame_id', 'xmin', 'ymin', 'xmax', 'ymax', 'classification'],
  )
  first_frame = result_df.loc[result_df.frame_id == 0]
  final_results = []
  previous_frame_objects = []
  # Get first frame's all objects for next tracking.
  for i in range(len(first_frame)):
    value = first_frame.iloc[i]
    previous_frame_objects.append([
        0,
        value['xmin'],
        value['ymin'],
        value['xmax'],
        value['ymax'],
        value['classification'],
        f'object_{i}',
    ])
  final_results.append(previous_frame_objects)
  num_objects = len(previous_frame_objects) - 1
  num_objects_in_frame = 0
  for i in range(1, len(frames)):
    new_frame_object_df = result_df.loc[result_df.frame_id == i]
    new_frame_objects = []
    exist_id = []
    for j in range(len(new_frame_object_df)):
      row = new_frame_object_df.iloc[j]
      pre_frame_index = -1
      new_appeared = False
      obj_id = -1

      while (pre_frame_index > -5) and (
          len(final_results) >= abs(pre_frame_index)
      ):
        previous_frame_objects = [
            obj
            for obj in final_results[pre_frame_index]
            if obj[-1] not in exist_id
        ]

        if len(previous_frame_objects) > 0:
          new_appeared, obj_id, _ = iou_calculate(
              row.tolist()[1:5], previous_frame_objects
          )
          if new_appeared:
            pre_frame_index -= 1
            continue
          else:
            break
        else:
          pre_frame_index -= 1
          new_appeared = True
          continue

      if new_appeared:
        obj_id = max(num_objects, num_objects_in_frame) + 1

      exist_id.append(f'object_{obj_id}')
      new_frame_objects.append([
          i,
          row['xmin'],
          row['ymin'],
          row['xmax'],
          row['ymax'],
          row['classification'],
          f'object_{obj_id}'
      ])
      num_objects_in_frame = max(num_objects_in_frame, obj_id)
    num_objects = num_objects_in_frame
    final_results.append(new_frame_objects)

  final_results = [itme for result in final_results for itme in result]
  final_df = pd.DataFrame(
      final_results,
      columns=[
          'frame',
          'xmin',
          'ymin',
          'xmax',
          'ymax',
          'classification',
          'object_id',
      ],
  )
  write_csv_file(f'{output_path}', final_df)
  print(f'[+] Video {input_video} done. Saved: {output_path}')


def read_csv_file(
    csv_path: str, header: str = 'infer', names: bool = None
) -> pd.DataFrame:
  with open(csv_path, 'r') as fh:
    if header:
      df = pd.read_csv(fh, header=header)
    else:
      df = pd.read_csv(fh, header=header, names=names)
  return df


def index_to_timestamp(index: int, fps: int) -> int:
  return int(round(1e6 * index / fps))


def read_video_csv(
    df: pd.DataFrame, file_id: str, image_base_path: str
) -> pd.DataFrame:
  total_frame = df.frame.max()
  print(f'[+] total frame {total_frame}')
  # class_id has to start from 1 for tf.example.
  class_name_to_id_map = dict(
      zip(_ITEMS_CLASS_TYPE, range(1, 1 + len(_ITEMS_CLASS_TYPE)))
  )

  df_dict = list()
  for _, row in df.iterrows():
    class_id = class_name_to_id_map[row.classification]
    image_path = f'{image_base_path}/{file_id}/{file_id.zfill(4)}_{str(row.frame).zfill(6)}.png'
    track_id = row.object_id.split('_')[-1]
    df_dict.append({
        'class_id': int(class_id),
        'class_name': row.classification,
        'timestamp': int(index_to_timestamp(row.frame, 30)),
        'path': image_path,
        'video_id': file_id,
        'frame_index': int(row.frame),
        'x_min': float(row.xmin),
        'y_min': float(row.ymin),
        'x_max': float(row.xmax),
        'y_max': float(row.ymax),
        'track_id': int(track_id),
        'track_name': row.object_id,
    })

  return pd.DataFrame(df_dict)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  input_video = FLAGS.input_video
  assert input_video, ('Template_path must exist!')

  output_csv = FLAGS.output_csv
  assert output_csv, ('Output_path must exist!')

  output_file_dir = FLAGS.output_file_dir
  assert output_csv, ('output_file_dir must exist!')

  run_track(
      input_video,
      output_csv,
      output_file_dir
  )


if __name__ == '__main__':
  app.run(main)
