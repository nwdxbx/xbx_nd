/data_b/Framework/tensorflow/bazel-bin/tensorflow/contrib/lite/toco/toco \
  --input_file=yolo_frozen.pb \
  --input_format=TENSORFLOW_GRAPHDEF --output_format=TFLITE \
  --output_file=yolo_float.tflite --inference_type=FLOAT \
  --input_arrays=input_images \
  --output_arrays=out --input_shapes=1,160,160,3 \
  --quantize_weights=true