"""
Tools for interactive segmentation
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import pyplot as plt


class InteractiveSegmentation:
    """
    Interactive segmentation tool. Opens a window from which you can click to record pixel positions.
    """

    def __init__(
        self,
        segmentation_model: str = "sam2",
        segmentation_model_cfg: dict = dict(
            checkpoint="sam2/checkpoints/sam2.1_hiera_large.pt",
            model_cfg="sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
        ),
    ):
        """
        Args:
            segmentation_model: The segmentation model to use. Currently only "sam2" is supported.
        """
        self.segmentation_model = segmentation_model
        if self.segmentation_model == "sam2":
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            model_cfg = segmentation_model_cfg["model_cfg"]
            checkpoint = segmentation_model_cfg["checkpoint"]
            self.predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
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

                    if self.segmentation_model == "sam2":
                        clicked_points_np = np.array(clicked_points)
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
