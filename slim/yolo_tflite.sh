/data_b/Framework/tensorflow/bazel-bin/tensorflow/contrib/lite/toco/toco \
--input_file=yolo_frozen.pb \
--output_file=yolo.tflite \
--input_format=TENSORFLOW_GRAPHDEF \
--output_format=TFLITE \
--inference_type=QUANTIZED_UINT8 \
--input_shape="1,160,160,3" \
--input_array=input_images \
--output_array=out \
--std_value=255.0 --mean_value=0.0
#--default_ranges_min=0 --default_ranges_max=1

#/data_b/tensorflow/bazel-bin/tensorflow/lite/toco/toco \
# /data_b/Framework/tensorflow/bazel-bin/tensorflow/contrib/lite/toco/toco \
# --input_file=yolo_frozen.pb \
# --output_file=yolo.tflite \
# --input_format=TENSORFLOW_GRAPHDEF \
# --output_format=TFLITE \
# --inference_type=QUANTIZED_UINT8 \
# --input_shape="1,160,160,3" \
# --input_array=input_images \
# --output_array=out \
# --std_value=255.0 --mean_value=0.0
