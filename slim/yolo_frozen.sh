/data_b/Framework/tensorflow/bazel-bin/tensorflow/python/tools/freeze_graph \
--input_graph=yolo_eval.pb \
--input_checkpoint=yolo.ckpt-000 \
--output_graph=yolo_frozen.pb \
--output_node_names=out
#--output_node_names=out
#pred/Conv2D