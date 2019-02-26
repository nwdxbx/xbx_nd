/data_b/Framework/tensorflow/bazel-bin/tensorflow/contrib/lite/toco/toco \
--input_file=frozen_landmark_eval.pb \
--output_file=landmark_eval.tflite \
--input_format=TENSORFLOW_GRAPHDEF \
--output_format=TFLITE \
--inference_type=QUANTIZED_UINT8 \
--input_shape="1,64,64,3" \
--input_array=input_images \
--output_array=logits/out \
--std_value=255.0 --mean_value=0.0 \
--default_ranges_min=0 --default_ranges_max=255


