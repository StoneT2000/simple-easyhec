"""
Tools for interactive segmentation
"""

from typing import Callable, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import pyplot as plt


class InteractiveSegmentation:
    """
    Interactive segmentation tool. Allows you to give a list of images to annotate through a CV2 window from which you can click to annotat positive/negative points.

    Args:
        segmentation_model: The segmentation model to use. Can be a string indicating a model that is already setup or a callable function to generate segmentation masks given point annotations and an image.

            The callable function should be of the form def `segment(image: np.ndarray, clicked_points_np: np.ndarray) -> np.ndarray`. Image is shape (H, W, 3) and clicked_points_np is shape (N, 3) where each element is a (x, y, label) tuple. The label is 1 for positive points and -1 for negative points. The function should return a numpy array of shape (H, W) where each pixel is either 0 or 1 for the mask.

        segmentation_model_cfg: The configuration for the segmentation model. Only used if the segmentation model is a string.
    """

    def __init__(
        self,
        segmentation_model: Union[
            str, Callable[[np.ndarray, np.ndarray], np.ndarray]
        ] = "sam2",
        segmentation_model_cfg: dict = dict(
            checkpoint="sam2/checkpoints/sam2.1_hiera_large.pt",
            model_cfg="sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
        ),
    ):

        self.segmentation_model = segmentation_model
        if self.segmentation_model == "sam2":
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            model_cfg = segmentation_model_cfg["model_cfg"]
            checkpoint = segmentation_model_cfg["checkpoint"]
            self.predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

            def segment(image, clicked_points_np):
                input_label = clicked_points_np[:, 2]
                input_point = clicked_points_np[:, :2]
                with torch.inference_mode(), torch.autocast(
                    "cuda", dtype=torch.bfloat16
                ):
                    self.predictor.set_image(image)
                    mask, _, _ = self.predictor.predict(
                        input_point, input_label, multimask_output=False
                    )
                    mask = mask[0]
                return mask

            self._segment = segment
        elif isinstance(self.segmentation_model, Callable):
            self._segment = self.segmentation_model
        else:
            raise ValueError(
                f"Segmentation model {self.segmentation_model} not supported"
            )

    def get_segmentation(self, images: np.ndarray):
        """
        Get segmentation from a list of imagees. Opens a window from which you can click to record points of the object to segment out.

        There are a few other options that let the user e.g. redo the segmentation, redo the points etc., see the terminal output for help
        """
        state = "annotation"
        current_image_idx = 0
        masks = []
        clicked_points = []
        state = "annotation"

        def print_help_message():
            print(
                f"Currently annotating image {current_image_idx+1}/{len(images)}. Click to add a point of what to segment, right click to add a negative point of what not to segment. Press 't' to generate a candidate segmentation mask. Press 'r' to clear the current point annotation. Press 'e' to edit the existing annotation points."
            )

        def mouse_callback(event, x, y, flags, param):
            nonlocal clicked_points
            if event == cv2.EVENT_LBUTTONDOWN:
                clicked_points.append((x, y, 1))
            elif event == cv2.EVENT_RBUTTONDOWN:
                clicked_points.append((x, y, -1))

        # Display the image and set mouse callback
        annotation_window_name = "Annotation: Click for positive points, right click for negative points. 'r' to reset, 'e' to edit, 't' to generate the segmentation"
        check_window_name = (
            "Check segmentation quality. Press 't' to proceed. Press 'e' to edit again."
        )
        cv2.namedWindow(annotation_window_name, cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback(annotation_window_name, mouse_callback)

        print_help_message()

        point_size = int(0.01 * (images[0].shape[0] + images[0].shape[1]) / 2)
        while current_image_idx < len(images):
            display_img = images[current_image_idx].copy()
            image = display_img.copy()
            key = cv2.waitKey(1)
            if state == "annotation":
                if clicked_points:
                    for x, y, label in clicked_points:
                        cv2.circle(
                            display_img,
                            (x, y),
                            point_size,
                            (25, 200, 25) if label == 1 else (200, 25, 25),
                            -1,
                        )
                if key == ord("r"):
                    print("(r)esetting the point annotations")
                    clicked_points = []
                elif key == ord("e"):
                    print("Entering (e)dit mode")
                elif key == ord("t"):
                    if len(clicked_points) == 0:
                        print(
                            "No points to generate the segmentation mask. Make sure to add at least one point."
                        )
                        continue
                    print(
                        "Generating the segmentation mask, check its quality. If the mask is good press 't' again to move on."
                    )
                    cv2.setWindowTitle(annotation_window_name, check_window_name)
                    state = "check"
                    clicked_points_np = np.array(clicked_points)
                    mask = self._segment(image, clicked_points_np)
                    state = "segmentation"
            elif state == "segmentation":
                mask_color = np.array([30, 144, 255])
                mask_overlay = mask.astype(float).reshape(
                    image.shape[0], image.shape[1], 1
                ) * mask_color.reshape(1, 1, -1)
                display_img = mask_overlay * 0.6 + display_img * 0.4
                display_img[mask == 0] = image[mask == 0]
                display_img = display_img.astype(np.uint8)
                if key == ord("t"):
                    masks.append(mask)
                    current_image_idx += 1
                    state = "annotation"
                    clicked_points = []
                    if current_image_idx < len(images):
                        print_help_message()
                elif key == ord("e"):
                    print("Entering (e)dit mode")
                    cv2.setWindowTitle(annotation_window_name, annotation_window_name)
                    state = "annotation"
                elif key == ord("r"):
                    print("(r)esetting the point annotations")
                    clicked_points = []
                    state = "annotation"
            cv2.imshow(
                annotation_window_name, cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)
            )
        cv2.destroyWindow(annotation_window_name)
        return np.stack(masks)
