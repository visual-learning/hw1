Please follow the instructions for this part of the assignment in THIS order!

0. Read the FCOS paper (Fully Convolutional One-stage Object Detection), and 
skim through the starter code. You will be implementing functions in `detection_utils.py`
and `one_stage_detector.py`, and you will be using `train.py` and `test_one_stage_detection.py`
extensively. To run the tests, you should run `python3 -m test_object_detection`. At this 
point, most tests should be skipped, and a couple will pass.

1. Follow the instructions in `one_stage_detector.py` to implement additional FPN
layers for transforming `(c3, c4, c5)` to `(p3, p4, p5)`. See Figure 2 in
the original FCOS paper for reference.

2. Complete the `__init__` and `forward()` methods of FCOSPredictionNetwork
in `one_stage_detector.py` . See the test cases to see
how this network will be used. In the expected output. The classification 
logits have `NUM_CLASSES` channels, box regression deltas have 4 output 
channels, and centerness has 1 output channel.
The height and width of all outputs are flattened to one dimension, resulting 
in `(B, H * W, C)` format - this format is more convenient for computing loss.
Now that the network is complete, we need to compute the ground truth 
data for training the network. We will need to assign a GT target to 
every prediction from the model.

3. Complete the function get_fpn_location_coords in `detection_utils.py` to get the 
x, y locations for all the FPN features for each level.
Confirm your implementation for (3) is correct by running the test suite. We have also implemented 
the matching of locations to GT boxes for you. To visualize these boxes in Tensorboard, run `python3 train.py --visualize_gt`

4. Next, implement `fcos_get_deltas_from_locations` and `fcos_apply_deltas_to_locations`
in `detection_utils.py`, to compute the box regression deltas. Use the test suite to verify
your implementation.

5. Implement the function `fcos_make_centerness_targets` in `detection_utils.py`. 
By this point, all the tests in the suite should be passing. You need the 
previous steps to work properly before moving on.

6. We have provided examples of how to call the loss functions in the 
test suite. Please read the loss-related test cases carefully in order
to understand them. We now need to combine everything into the FCOS class
in this file. Use the unit tests as a reference and implement the loss computation
in the `forward()` function of `FCOS`. We have already done most of the heavy lifting, 
you should just need to combine your parts together!

7. Now that your loss should be computed, run `train.py` with the overfit option
enabled: `python3 train.py --overfit`. 
This will check if the network is able to fit to a small dataset. If 
everything is correct, the loss will increase at first (due to exponential moving averaging), 
but should go down after that. Once you have verified your implementation is correct, please
submit the training loss curve plot in the PDF submission.

8. Next, train a full network, with the command
`python3 train.py --overfit=False`.
 With the default hyperparameters, this should
take ~30 minutes to train. Use `tmux` to run the train.py script so you do not
need to watch it constantly. At this point, you will need a GPU, so you
should train this part on AWS. Please submit the trainig loss curve plot in your PDF submission.

9. Follow the instructions in `FCOS.inference` to implement inference. We have 
implemented a function to run NMS for you. Test your inference implementation
by turning on the `inference` and `test_inference` flags in `train.py`. This will
upload some sample detections to TensorBoard. The results should be generally accurate.

10. Finally, run `python3 train.py --inference --test_inference=False` to compute 
the final mAP. If everything is correct, you should obtain >= 20 mAP. The inference
method will output a file with the class-wise mAP and the overall mAP. Please
submit this in your PDF submission.
