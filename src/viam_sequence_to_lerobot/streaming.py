"""lerobot dataset writer that never drops a frame.

Imported lazily: importing lerobot installs a root logging handler, which
would silence the CLI's own logging setup if it happened at module load.
"""

from __future__ import annotations

import time

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import StreamingVideoEncoder


class BlockingStreamingEncoder(StreamingVideoEncoder):
    """Wait for queue space instead of dropping frames.

    lerobot's encoder drops a frame (with a warning) when its queue stays full
    for 100 ms, which keeps a live recording loop responsive. Offline conversion
    has no such deadline, and a dropped frame would desynchronise video from
    state, so the producer blocks instead.
    """

    def feed_frame(self, video_key: str, image: np.ndarray) -> None:
        queue = self._frame_queues[video_key]
        while queue.full() and self._threads[video_key].is_alive():
            time.sleep(0.005)
        super().feed_frame(video_key, image)


class BlockingDataset(LeRobotDataset):
    @staticmethod
    def _build_streaming_encoder(
        fps, rgb_encoder, depth_encoder, encoder_queue_maxsize, encoder_threads
    ) -> StreamingVideoEncoder:
        return BlockingStreamingEncoder(
            fps=fps,
            rgb_encoder=rgb_encoder,
            depth_encoder=depth_encoder,
            queue_maxsize=encoder_queue_maxsize,
            encoder_threads=encoder_threads,
        )
