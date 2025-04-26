"""Library with util functions for handling video."""

import tempfile
import shutil

import cv2
import numpy as np
from PIL import Image
import skimage.util
import tensorflow as tf


Box = tuple[int, int, int, int]
BoxNorm = tuple[float, float, float, float]


class DetectionTFModel:
  """Class to load tflite models and use them in inference mode.

    Args:
      tflite_model_path: path to saved tf_lite model.
      detection_threshold: confidence threshold required to accept a detection.
  """

  def __init__(self, tflite_model_path: str, detection_threshold: float = 0.15):
    self.interpreter = self._load_tflite_model(tflite_model_path)
    self._input_details = self.interpreter.get_input_details()
    self._output_details = self.interpreter.get_output_details()
    self._detection_threshold = detection_threshold

  def infer(self, frame: np.ndarray) -> list[tuple[BoxNorm, str, float]]:
    """Performs an inference step on a single frame."""
    image = resize_np_image(frame, self._get_shape_list())
    self.interpreter.set_tensor(self._input_details[0]['index'], image)
    self.interpreter.invoke()
    boxes = self._get_output_by_index(0)
    names = self._get_output_by_index(1)
    confidences = self._get_output_by_index(2)
    output = zip(boxes, names, confidences)
    return [(box, name, confidence) for box, name, confidence in output
            if confidence > self._detection_threshold]

  def _get_output_by_index(self, index: int):
    output_details = self._output_details[index]['index']
    return self.interpreter.get_tensor(output_details)[0].tolist()

  def _get_shape_list(self) -> list[int]:
    return self._input_details[0].get('shape').tolist()

  @staticmethod
  def _load_tflite_model(model_path: str) -> tf.lite.Interpreter:
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
    image = skimage.util.img_as_ubyte(image)
  return image


def box_norm_to_box(box_norm: BoxNorm, frame: np.ndarray) -> Box:
  ymin, xmin, ymax, xmax = box_norm
  y1, x1, y2, x2 = (
      int(ymin * frame.shape[0]),
      int(xmin * frame.shape[1]),
      int(ymax * frame.shape[0]),
      int(xmax * frame.shape[1]),
  )
  return (y1, x1, y2, x2)


def box_to_cropped_image(box: Box, frame: np.ndarray) -> np.ndarray:
  y1, x1, y2, x2 = box
  return frame[y1:y2, x1:x2]


def box_area(box: Box) -> float:
  y1, x1, y2, x2 = box
  return abs(x2 - x1 + 1) * abs(y2 - y1 + 1)


def compute_intersection_box(box1: Box, box2: Box) -> Box:
  """Computes the intersection box between two boxes."""
  y11, x11, y12, x12 = box1
  y21, x21, y22, x22 = box2
  y1 = max(y11, y21)
  x1 = max(x11, x21)
  y2 = min(y12, y22)
  x2 = min(x12, x22)
  return (y1, x1, y2, x2)


def compute_iou(box1: Box, box2: Box) -> float:
  """Computes the IOU between two boxes."""
  box1_area = box_area(box1)
  box2_area = box_area(box2)
  intersection_area = box_area(compute_intersection_box(box1, box2))
  iou = intersection_area / (box1_area + box2_area - intersection_area)
  return iou


def video_to_frames(video_path: str) -> list[np.ndarray]:
  """Loads video using OpenCV."""
  frame_list = []
  with tempfile.NamedTemporaryFile() as f:
    shutilcopy(video_path, f.name)
    capture = cv2.VideoCapture(f.name)
    video_length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    start = 0
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    _, frame = capture.read()
    frame_list.append(frame)
    for i in range(video_length - 1):
      ret, frame = capture.read()
      if not ret:
        raise ValueError(f'Unable to read frame number {i} at: {video_path}')
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      frame_list.append(frame)
    capture.release()
  return frame_list


def mask_to_slices(mask: list[bool]) -> list[tuple[int, int]]:
  mask.insert(0, False)
  mask.append(False)  # Padding for edge cases.
  slices = list()
  start = 1
  for i in range(1, len(mask)):
    if mask[i] and not mask[i - 1]:
      start = i - 1
    elif not mask[i] and mask[i - 1]:
      end = i - 1
      slices.append((start, end))
  return slices


def fix_mask(mask: list[bool], tolerance: int = 0) -> list[bool]:
  """Fix gaps of missing boxes based on preset tolerance."""
  missing_box_count = tolerance
  for i in range(len(mask)):
    is_box = mask[i]
    if not is_box:
      missing_box_count += 1
      continue
    if missing_box_count <= tolerance:
      while missing_box_count:
        mask[i - missing_box_count] = True
        missing_box_count -= 1
    else:
      missing_box_count = 0
  return mask


def slice_frames(
    frames: list[np.ndarray],
    tolerance: int,
    path_to_model: str,
) -> list[list[np.ndarray]]:
  """Selects the start and end indices of frame slices with boxes.

  Args:
    frames: list of frames
    tolerance: tolerance on how many frames can go without boxes before starting
    a new slice.
    path_to_model: path to detection model

  Returns:
    a list of start and end indices for frame slices with boxes.
  """
  mask = [
      n_boxes > 0
      for n_boxes in n_boxes_in_frames(frames, path_to_model)
  ]
  slices = mask_to_slices(fix_mask(mask, tolerance=tolerance))
  return [frames[start:end] for start, end in slices]


def filter_overlapping_boxes(
    boxes: list[Box],
    confidences: list[float],
    overlap_threshold: float = 0.4,
) -> list[Box]:
  """Remove the overlapping boxes from a list of boxes, based on confidence.

  Args:
    boxes: list of boxes
    confidences: boxes confidence list
    overlap_threshold: overlap thresholds

  Returns:
    non-overlapping boxes.
  """
  sorted_boxes, _ = zip(
      *sorted(list(zip(boxes, confidences)), key=lambda x: x[1], reverse=True)
  )
  i = 0
  while i < len(sorted_boxes):
    len_boxes = len(sorted_boxes)
    j = i + 1
    while j < len_boxes:
      iou = compute_iou(sorted_boxes[i], sorted_boxes[j])
      if iou > overlap_threshold:
        boxes.pop(j)
      else:
        j += 1
    i += 1
  return boxes


def detect_objects(
    frames: list[np.ndarray], path_to_detection_model: str
) -> list[list[Box]]:
  """Detects items in a sequence of frames."""
  tf_model = DetectionTFModel(path_to_detection_model)
  item_id = 1  # The model consumes various ids for different items.

  def detect_objects_in_frame(frame: np.ndarray) -> list[Box]:
    """Detects items in a single frame."""
    output = tf_model.infer(frame)
    if not output:
      return []
    boxes_norm, names, confidences = zip(*output)
    item_boxes_norm = [
        box for name, box in zip(names, boxes_norm) if name == item_id
    ]
    item_boxes = [
        box_norm_to_box(box_norm, frame) for box_norm in item_boxes_norm
    ]
    return filter_overlapping_boxes(
        item_boxes, confidences, overlap_threshold=0.4
    )

  return [detect_objects_in_frame(frame) for frame in frames]


def n_boxes_in_frames(
    frames: list[np.ndarray], path_to_detection_model: str
) -> list[int]:
  return [
      len(boxes)
      for boxes in detect_objects(
          frames, path_to_detection_model=path_to_detection_model
      )
  ]


def save_frames(frames: list[np.ndarray], image_base_path: str,
                file_id: str) -> None:
  """Saves frames on the provided image_base_path with the name file_id."""
  for count, frame in enumerate(frames):
    img_pil = Image.fromarray(np.uint8(frame))
    img_save_path = (
        f'{image_base_path}/{file_id}/{file_id.zfill(4)}_{str(count).zfill(6)}.png'
    )
    with open(img_save_path, 'wb') as f:
      img_pil.save(f, 'PNG')


def generate_sequence_example(
    class_id: list[int],
    class_name: list[str],
    timestamp: int,
    x_min: list[float],
    y_min: list[float],
    x_max: list[float],
    y_max: list[float],
    video_id: str,
    frame_index: int,
    track_id: list[int],
    track_name: list[str],
    dataset_name: str,
    path: str | None = None,
    encoded_image: bytes | None = None,
) -> tf.train.SequenceExample:
  """Generate tf SequenceExample."""
  example = tf.train.SequenceExample()
  example.context.feature['clip/start/timestamp'].int64_list.value.append(0)
  example.context.feature['clip/key_frame/timestamp'].int64_list.value.append(
      timestamp
  )
  example.context.feature['clip/key_frame/frame_index'].int64_list.value.append(
      frame_index
  )
  if path:
    example.context.feature['clip/data_path'].bytes_list.value.append(
        bytes(path, 'utf-8')
    )
  else:
    if not encoded_image:
      raise ValueError('Provide either path to image or encoded image')
    example.context.feature['encoded_image'].bytes_list.value.append(
        encoded_image
    )
  example.context.feature['video_id'].bytes_list.value.append(
      bytes(video_id, 'utf-8')
  )
  example.context.feature['dataset_name'].bytes_list.value.append(
      bytes(dataset_name, 'utf-8')
  )

  # Adds box annotations.
  example.context.feature[
      'clip/key_frame/bbox/label/string'
  ].bytes_list.value.extend([bytes(label, 'utf-8') for label in class_name])
  example.context.feature[
      'clip/key_frame/bbox/label/index'
  ].int64_list.value.extend(class_id)
  example.context.feature['clip/key_frame/bbox/xmin'].float_list.value.extend(
      x_min
  )
  example.context.feature['clip/key_frame/bbox/xmax'].float_list.value.extend(
      x_max
  )
  example.context.feature['clip/key_frame/bbox/ymin'].float_list.value.extend(
      y_min
  )
  example.context.feature['clip/key_frame/bbox/ymax'].float_list.value.extend(
      y_max
  )
  example.context.feature['region/track/index'].int64_list.value.extend(
      track_id
  )
  example.context.feature['region/track/string'].bytes_list.value.extend(
      [bytes(name, 'utf-8') for name in track_name]
  )
  return example
