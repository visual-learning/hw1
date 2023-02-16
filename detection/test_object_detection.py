import unittest
import torch
from torchvision.ops import sigmoid_focal_loss
import torch.nn.functional as F

from detection_utils import *
from one_stage_detector import *
from utils.grad import rel_error

"""
This file provides tests for several of the functions you will be implementing. 
As a reminder, this file is not completely exhaustive, and passing all these tests
will not necessarily guarantee that everything is correct. You may need to debug
issues with visualization, or even writing your own tests. In other words, 
passing these tests is necessary, but not necessarily sufficient for a correct
implementation.

Run the test suite with : 
    python3 -m test_object_detection
"""

class TestDetectorBackboneWithFPN(unittest.TestCase):
    def setUp(self):
        self.out_channels = 64
        self.num_classes = 20
        if torch.cuda.is_available():
            #print("Using CUDA for testing.")
            self.device = torch.device("cuda")
        else:
            #print("Using CPU, CUDA not available for testing.")
            self.device = torch.device("cpu")

        self.backbone = DetectorBackboneWithFPN(out_channels=self.out_channels).to(self.device)
        dummy_in = torch.randn(2, 3, 224, 224).to(self.device)
        self.dummy_fpn_feats = self.backbone(dummy_in)

    def test_backbone(self):
        if self.dummy_fpn_feats['p3'] is None:
            self.skipTest("Detector backbone not implemented.")

        self.assertEqual(self.dummy_fpn_feats["p3"].shape, torch.Size([2, self.out_channels, 28, 28]))
        self.assertEqual(self.dummy_fpn_feats["p4"].shape, torch.Size([2, self.out_channels, 14, 14]))
        self.assertEqual(self.dummy_fpn_feats["p5"].shape, torch.Size([2, self.out_channels, 7, 7]))

    def test_prediction_network(self):
        self.pred_net = FCOSPredictionNetwork(
            num_classes=self.num_classes, in_channels=64, stem_channels=[64, 64]).to(self.device)
        if self.pred_net.pred_cls is None:
            self.skipTest("FCOSPredictionNetwork is not set up yet.")

        class_logits, boxreg_deltas, centerness_logits = self.pred_net(self.dummy_fpn_feats)
        if len(class_logits) == 0:
            self.skipTest("Detector backbone not implemented.")

        self.assertEqual(class_logits["p3"].shape, torch.Size([2, 784, self.num_classes]))
        self.assertEqual(boxreg_deltas["p3"].shape, torch.Size([2, 784, 4]))
        self.assertEqual(centerness_logits["p3"].shape, torch.Size([2, 784, 1]))

        self.assertEqual(class_logits["p4"].shape, torch.Size([2, 196, self.num_classes]))
        self.assertEqual(boxreg_deltas["p4"].shape, torch.Size([2, 196, 4]))
        self.assertEqual(centerness_logits["p4"].shape, torch.Size([2, 196, 1]))

        self.assertEqual(class_logits["p5"].shape, torch.Size([2, 49, self.num_classes]))
        self.assertEqual(boxreg_deltas["p5"].shape, torch.Size([2, 49, 4]))
        self.assertEqual(centerness_logits["p5"].shape, torch.Size([2, 49, 1]))

    def test_get_fpn_location_coords(self):
        if self.dummy_fpn_feats['p3'] is None:
            self.skipTest("Detector backbone not implemented.")
        fpn_feats_shapes = {
            level_name: feat.shape for level_name, feat in self.dummy_fpn_feats.items()
        }

        # Get CPU tensors for this sanity check: (you can pass `device=` argument.
        locations_per_fpn_level = get_fpn_location_coords(fpn_feats_shapes, self.backbone.fpn_strides)
        expected_locations = {
            "p3": torch.tensor([[4.0, 4.0], [4.0, 12.0], [4.0, 20.0], [4.0, 28.0], [4.0, 36.0]]),
            "p4": torch.tensor([[8.0, 8.0], [8.0, 24.0], [8.0, 40.0], [8.0, 56.0], [8.0, 72.0]]),
            "p5": torch.tensor([[16.0, 16.0], [16.0, 48.0], [16.0, 80.0], [16.0, 112.0], [16.0, 144.0]]),
        }

        for level_name, locations in locations_per_fpn_level.items():
            if locations is None:
                self.skipTest("get_fpn_location_coords is not implemented.")
            error = rel_error(expected_locations[level_name], locations[:5, :])
            self.assertAlmostEqual(error, 0.0, places=5)
    
    def test_gt_targets_for_box_regression(self):
        input_boxes = torch.Tensor(
            [[10, 15, 100, 115, 1], [30, 20, 40, 30, 1], [120, 100, 200, 200, 1]]
        )
        input_locations = torch.Tensor([[30, 40], [32, 29], [125, 150]])

        # Here we do a simple sanity check - getting deltas for a particular set of boxes
        # and applying them back to centers should give us the same boxes. Setting a random
        # stride = 8, it should not affect reconstruction if it is same on both sides.
        _deltas = fcos_get_deltas_from_locations(input_locations, input_boxes, stride=8)
        if _deltas is None:
            self.skipTest("fcos_get_deltas_from_test is not implemented.")
        output_boxes = fcos_apply_deltas_to_locations(_deltas, input_locations, stride=8)
        box_error = rel_error(input_boxes[:, :4], output_boxes)
        self.assertEqual(box_error, 0.0)

    def test_gt_deltas_invalid_box(self):
        # Another check: deltas for GT class label = -1 should be -1.
        background_box = torch.Tensor([[-1, -1, -1, -1, -1]])
        input_location = torch.Tensor([[100, 200]])
        _deltas = fcos_get_deltas_from_locations(input_location, background_box, stride=8)
        if _deltas is None:
            self.skipTest("FCOS get deltas_from test is not implemented.")
        output_box = fcos_apply_deltas_to_locations(_deltas, input_location, stride=8)
        
        self.assertTrue(torch.equal(_deltas, torch.tensor([[-1., -1., -1., -1.]])))
        self.assertTrue(torch.equal(output_box, torch.tensor([[100., 200., 100., 200.]])))

    def test_make_centerness_targets(self):
        input_boxes = torch.Tensor(
            [
                [10, 15, 100, 115, 1],
                [30, 20, 40, 30, 1],
                [-1, -1, -1, -1, -1]  # background
            ]
        )
        input_locations = torch.Tensor([[30, 40], [32, 29], [125, 150]])
        expected_centerness = torch.Tensor([0.30860671401, 0.1666666716, -1.0])
        _deltas = fcos_get_deltas_from_locations(input_locations, input_boxes, stride=8)
        if _deltas is None:
            self.skipTest("fcos_get_deltas_from_test is not implemented.")
        centerness = fcos_make_centerness_targets(_deltas)
        self.assertAlmostEqual(rel_error(centerness, expected_centerness), 0.0, places=5)

    def test_sigmoid_focal_loss(self):
        dummy_pred_cls_logits = torch.tensor([[[-0.0139, -0.7245, -0.5730, -1.7872,  1.5967],
         [-0.9421, -0.2319, -0.7095, -0.0868, -0.2936]]])
        # Corresponding one-hot vectors of GT class labels (2, -1), one
        # foreground and one background.
        # shape: (batch_size, num_locations, num_classes)
        dummy_gt_classes = torch.Tensor([[[0, 0, 1, 0, 0], [0, 0, 0, 0, 0]]])

        # This loss expects logits, not probabilities (DO NOT apply sigmoid!)
        cls_loss = sigmoid_focal_loss(
            inputs=dummy_pred_cls_logits, targets=dummy_gt_classes)
        self.assertAlmostEqual(cls_loss.sum().cpu().numpy(), 1.514438, places=3)

    def test_centerness_and_regression_loss(self):
        # STUDENTS: use this example as a reference for how to 
        # use the loss functions.
        dummy_locations = torch.Tensor([[32, 32], [64, 32]])
        dummy_gt_boxes = torch.Tensor(
            [
                [1, 2, 40, 50, 2],
                [-1, -1, -1, -1, -1]  # Same GT classes as above cell.
            ]
        )
        # shape: (batch_size, num_locations, 4 or 1)
        dummy_pred_boxreg_deltas = torch.Tensor(
            [[[ 0.1108, -0.2993, -0.6284,  0.8650],
            [ 0.5013,  0.5564, -2.0083,  0.4059]]])

        dummy_pred_ctr_logits = torch.Tensor(
            [[[-1.1266],
            [ 0.0109]]])

        # Collapse batch dimension.
        dummy_pred_boxreg_deltas = dummy_pred_boxreg_deltas.view(-1, 4)
        dummy_pred_ctr_logits = dummy_pred_ctr_logits.view(-1)

        # First calculate box reg loss, comparing predicted boxes and GT boxes.
        dummy_gt_deltas = fcos_get_deltas_from_locations(
            dummy_locations, dummy_gt_boxes, stride=32
        )
        if dummy_gt_deltas is None:
            self.skipTest("fcos_get_deltas_from_test is not implemented.")
        else:
            diff_mean = (dummy_gt_deltas - torch.tensor([[ 0.9688,  0.9375,  0.2500,  0.5625],
                            [-1.0000, -1.0000, -1.0000, -1.0000]])).mean()
            self.assertAlmostEqual(diff_mean.cpu().numpy(), 0, places=4)

            # Multiply with 0.25 to average across four LTRB components.
            loss_box = 0.25 * F.l1_loss(
                dummy_pred_boxreg_deltas, dummy_gt_deltas, reduction="none"
            )
            # No loss for background:
            loss_box[dummy_gt_deltas < 0] *= 0.0
            diff_mean = (loss_box - torch.tensor([[0.2145, 0.3092, 0.2196, 0.0756],
                                    [0.0000, 0.0000, 0.0000, 0.0000]])).mean()
            self.assertAlmostEqual(diff_mean.cpu().numpy(), 0, places=4)

        # Now calculate centerness loss.
         # Centerness is just a dummy value:
        dummy_gt_centerness = torch.tensor([0.6, -1])
        centerness_loss = F.binary_cross_entropy_with_logits(
            dummy_pred_ctr_logits, dummy_gt_centerness, reduction="none"
        )
        # No loss for background:
        centerness_loss[dummy_gt_centerness < 0] *= 0.0
        diff_mean = (centerness_loss - torch.tensor([0.9567, 0.0000])).mean()
        self.assertAlmostEqual(diff_mean.cpu().numpy(), 0, places=4)
    
if __name__ == '__main__':
    unittest.main()