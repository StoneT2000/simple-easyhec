"""
Tools for interactive segmentation
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import pyplot as plt


class ImageRenderer:
    def __init__(self, wait_for_button_press=True):
        """
        Create a very light-weight image renderer.

        Args:
            wait_for_button_press (bool): If True, each call to this renderer will pause the process until the user presses any key.
            event_handler: Code to run given an event / button press. If None the default is mapping 'escape' and 'q' to sys.exit(0)
        """
        self._image = None
        self.last_event = None
        self.wait_for_button_press = wait_for_button_press
        self.pressed_keys = set()

    def key_press_handler(self, event):
        self.last_event = event
        self.pressed_keys.add(event.key)
        if event.key in ["q", "escape"]:
            sys.exit(0)

    def key_release_handler(self, event):
        if event.key in self.pressed_keys:
            self.pressed_keys.remove(event.key)

    def __call__(self, buffer):
        if not self._image:
            plt.ion()
            self.fig, self.ax = plt.subplots()
            self._image = self.ax.imshow(buffer, animated=True)
            self.fig.canvas.mpl_connect("key_press_event", self.key_press_handler)
            self.fig.canvas.mpl_connect("key_release_event", self.key_release_handler)
        else:
            self._image.set_data(buffer)
        if self.wait_for_button_press:
            plt.waitforbuttonpress()
        else:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        plt.draw()

    def __del__(self):
        self.close()

    def close(self):
        plt.ioff()
        plt.close()


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
        renderer = ImageRenderer(wait_for_button_press=False)
        state = "annotation"
        current_image_idx = 0
        masks = []
        annotation_objs = []
        clicked_points = []

        def print_help_message():
            if state == "annotation":
                print(
                    f"Currently annotating image {current_image_idx+1}/{len(images)}. Click to add a point of what to segment, right click to add a negative point of what not to segment. Press 't' when done. Press 'r' to clear the current point annotation and redo the points"
                )
            elif state == "segmentation":
                print(
                    f"Currently showing the predicted segmentation for image {current_image_idx+1}/{len(images)}. Press 't' to move on to the next image. Press 'e' to delete this segmentation and edit the existing annotation points. Press 'r' to delete this segmentation and re-annotate the points for this image."
                )

        def onclick(event):
            nonlocal annotation_objs, clicked_points
            if event.xdata is not None and event.ydata is not None:
                x, y = int(event.xdata), int(event.ydata)
                if event.button == 3:
                    clicked_points.append((x, y, 0))
                    annotation_objs.append(plt.plot(x, y, "ro")[0])
                else:
                    if x < 0 or x >= image.shape[1] or y < 0 or y >= image.shape[0]:
                        return
                    clicked_points.append((x, y, 1))
                    annotation_objs.append(plt.plot(x, y, "go")[0])

        def clear_drawn_points():
            nonlocal annotation_objs
            for x in annotation_objs:
                x.remove()
            annotation_objs = []

        renderer(images[0])
        renderer.ax.axis("off")
        cid = None
        print(
            f"Starting annotation process for {len(images)} images. Press 't' to finish annotation, 'r' to redo annotation. Press 'h' for help."
        )
        print("--------------------------------")
        print_help_message()
        while current_image_idx < len(images):
            image = images[current_image_idx].copy()
            key = renderer.last_event.key if renderer.last_event is not None else None
            if renderer.last_event is not None:
                renderer.last_event = None
            if key == "q":
                renderer.close()
                return None
            if key == "h":
                print_help_message()
            if state == "annotation":
                cid = renderer.fig.canvas.mpl_connect("button_press_event", onclick)
                renderer.ax.set_title(
                    "Click on the image to record annotation points for segmentation"
                )
                renderer(image)

                if key == "t":
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
                    clear_drawn_points()
                elif key == "r":
                    clear_drawn_points()
                    clicked_points = []
                    print("Cleared previous points")
            elif state == "segmentation":
                renderer.fig.canvas.mpl_disconnect(cid)
                masked_image = image.copy()
                mask_color = np.array([30, 144, 255])
                mask_overlay = mask.astype(float).reshape(
                    image.shape[0], image.shape[1], 1
                ) * mask_color.reshape(1, 1, -1)
                masked_image = mask_overlay * 0.6 + masked_image * 0.4
                masked_image[mask == 0] = image[mask == 0]
                renderer.ax.set_title("Check the segmentation quality")
                renderer(masked_image.astype(np.uint8))
                if key == "t":
                    masks.append(mask)
                    current_image_idx += 1
                    state = "annotation"
                    clear_drawn_points()
                    clicked_points = []
                    if current_image_idx < len(images):
                        print_help_message()
                elif key == "e":
                    state = "annotation"
                    # redraw existing points since they got removed to show the segmentation image
                    for x in annotation_objs:
                        x.remove()
                    annotation_objs = []
                    for pos in clicked_points:
                        annotation_objs.append(
                            renderer.ax.plot(
                                pos[0], pos[1], "ro" if pos[2] == 0 else "go"
                            )[0]
                        )
                elif key == "r":
                    state = "annotation"
                    clicked_points = []
                    clear_drawn_points()
                    print("Cleared previous points")
        renderer.close()
        return np.stack(masks)
