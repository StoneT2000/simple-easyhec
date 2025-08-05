# import torch
# from sam2.build_sam import build_sam2
# from sam2.sam2_image_predictor import SAM2ImagePredictor
# import numpy as np
# def show_mask(mask, ax, random_color=False, borders = True):
#     if random_color:
#         color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
#     else:
#         color = np.array([30/255, 144/255, 255/255, 0.6])
#     h, w = mask.shape[-2:]
#     mask = mask.astype(np.uint8)
#     mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
#     if borders:
#         import cv2
#         contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#         # Try to smooth contours
#         contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
#         mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
#     ax.imshow(mask_image)

# def show_points(coords, labels, ax, marker_size=375):
#     pos_points = coords[labels==1]
#     neg_points = coords[labels==0]
#     ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
#     ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

# def show_box(box, ax):
#     x0, y0 = box[0], box[1]
#     w, h = box[2] - box[0], box[3] - box[1]
#     ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

# def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
#     for i, (mask, score) in enumerate(zip(masks, scores)):
#         plt.figure(figsize=(10, 10))
#         plt.imshow(image)
#         show_mask(mask, plt.gca(), borders=borders)
#         if point_coords is not None:
#             assert input_labels is not None
#             show_points(point_coords, input_labels, plt.gca())
#         if box_coords is not None:
#             # boxes
#             show_box(box_coords, plt.gca())
#         if len(scores) > 1:
#             plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
#         plt.axis('off')
#         plt.show()

# checkpoint = "../sam2/checkpoints/sam2.1_hiera_large.pt"
# model_cfg = "../sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
# predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

# import gymnasium as gym
# import mani_skill.envs
# env = gym.make("StackCube-v1", shader_dir="rt")
# image = env.render_rgb_array().cpu().numpy()[0]
# # import ipdb; ipdb.set_trace()

# # with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
# #     predictor.set_image(image)
# #     masks, _, _ = predictor.predict(["robot"])
# # print(masks)
# import matplotlib.pyplot as plt

# clicked_points = []

# def onclick(event):
#     if event.xdata is not None and event.ydata is not None:
#         x, y = int(event.xdata), int(event.ydata)
#         clicked_points.append((x, y))
#         # print(f"Clicked at: ({x}, {y})")
#         # Optionally, plot a marker
#         plt.plot(x, y, 'ro')
#         plt.draw()

# fig, ax = plt.subplots()
# ax.imshow(image)
# cid = fig.canvas.mpl_connect('button_press_event', onclick)
# plt.title("Click on the image to record pixel positions. Press 'q' when done.")
# plt.show()


# input_point = np.array(clicked_points)
# input_label = np.array([1] * len(clicked_points))

# with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
#     predictor.set_image(image)
#     masks, scores, logits = predictor.predict(input_point, input_label, multimask_output=False)
# import ipdb; ipdb.set_trace()
# show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)
