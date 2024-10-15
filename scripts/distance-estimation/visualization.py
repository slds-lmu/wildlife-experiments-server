import os
import multiprocessing
from queue import Queue, Empty
import atexit
import numpy as np
import cv2


process_queue_max_len = min(8, os.cpu_count())
process_queue = Queue(process_queue_max_len)


def exit_handler():
    while not process_queue.empty():
        try:
            process = process_queue.get(block=False)
            process.join()
        except Empty:
            pass

atexit.register(exit_handler)


# prevents memory leaks by matplotlib
def visualize_farthest_calibration_frame(*args, **kwargs):
    if process_queue.full():
        oldest_process = process_queue.get()
        oldest_process.join()

    process = multiprocessing.Process(target=visualize_farthest_calibration_frame_impl, args=args, kwargs=kwargs)
    process.start()
    process_queue.put(process)


# prevents memory leaks by matplotlib
def visualize_detection(*args, **kwargs):
    if process_queue.full():
        oldest_process = process_queue.get()
        oldest_process.join()

    process = multiprocessing.Process(target=visualize_detection_impl, args=args, kwargs=kwargs)
    process.start()
    process_queue.put(process)


def visualize_farthest_calibration_frame_impl(data_dir, transect_id, farthest_calibration_frame_disp, min_depth, max_depth):
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use("pdf")  # required for PyInstaller detection
    plt.imshow(
        np.clip(
            farthest_calibration_frame_disp.data, max_depth ** -1, min_depth ** -1
        )
        ** -1,
        vmin=min_depth,
        vmax=max_depth,
        cmap="turbo",
    )
    plt.colorbar()
    plt.title("Final Calibration Frame Depth")
    os.makedirs(os.path.join(data_dir, "results", "calibration"), exist_ok=True)
    plt.savefig(
        os.path.join(data_dir, "results", "calibration", f"{transect_id}_calibration.pdf"),
        bbox_inches="tight",
        dpi=300,
        transparent=True,
    )


def visualize_detection_impl(data_dir, detection_id, detection_frame, calibrated_depth_midas, farthest_calibration_frame_disp, boxes, masks, world_positions, sample_locations, draw_detection_ids, draw_world_position, min_depth, max_depth):
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches
    matplotlib.use("pdf")  # required for PyInstaller detection
    scale = 2
    if farthest_calibration_frame_disp is not None:
        fig, (ax1, ax2, ax3) = plt.subplots(
            1, 3, figsize=(scale * 6.202, scale * 1.5)
        )
    else:
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(scale * 6.202, scale * 1.5)
        )
    
    ax1.imshow(detection_frame[..., ::-1])
    ax1.set_title("Observation")
    for i, (box, mask, world_pos) in enumerate(zip(boxes, masks, world_positions)):
        if mask is not None:
            mask_rgb = np.zeros((*detection_frame.shape[0:2], 4))
            mask_rgb[:, :, [0, 3]] = mask[..., None]
            mask_rgb[..., 3] *= 0.25
            ax1.imshow((mask_rgb * 255).astype(np.uint8))
        rect = matplotlib.patches.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax1.add_patch(rect)
        label = ""
        if draw_detection_ids:
            label += f"#{i:03d}"
        if draw_world_position:
            label += "@" + ",".join([f"{e:.2f}m" for e in world_pos])
        rx, ry = rect.get_xy()
        ax1.annotate(label, (rx, ry - 5), color="red", fontsize=6)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    im = ax2.imshow(calibrated_depth_midas, vmin=min_depth, vmax=max_depth, cmap="turbo")
    ax2.set_title("Depth")
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    sample_locations = np.array(sample_locations)
    if len(sample_locations) > 0:
        for ax in [ax1, ax2]:
            ax.scatter(
                sample_locations[:, 1],
                sample_locations[:, 0],
                color="r",
                marker="x",
            )
    if farthest_calibration_frame_disp is not None:
        im = ax3.imshow(
            np.clip(
                farthest_calibration_frame_disp.data,
                max_depth ** -1,
                min_depth ** -1,
            )
            ** -1,
            vmin=min_depth,
            vmax=max_depth,
            cmap="turbo",
        )
        ax3.set_title("Reference Depth")
        ax3.get_xaxis().set_visible(False)
        ax3.get_yaxis().set_visible(False)

    fig.subplots_adjust(
        bottom=0.1, top=0.9, left=0.1, right=0.8, wspace=0.02, hspace=0.02
    )
    a = 0.50
    cbar_ax = fig.add_axes([0.805, (1 - a) / 2, 0.02, a])
    cbar = fig.colorbar(im, cax=cbar_ax, ax=[ax2, ax3] if farthest_calibration_frame_disp is not None else [ax2])
    cbar.set_label("Depth [m]")

    os.makedirs(os.path.join(data_dir, "results", "sampling"), exist_ok=True)
    plt.savefig(
        os.path.join(data_dir, "results", "sampling", detection_id + ".pdf"),
        bbox_inches="tight",
        dpi=300,
        transparent=True,
    )