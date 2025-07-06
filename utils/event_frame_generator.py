"""
Event Frame Generation Module for Dual-Channel EventSR

This module converts raw event data into accumulated event frames for CNN processing.
Supports multiple accumulation strategies: time-based, count-based, and adaptive.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Union, List
from slayerSNN.spikeFileIO import event


class EventFrameGenerator:
    """
    Converts event streams into accumulated event frames for CNN processing.
    
    This class provides multiple strategies for accumulating events into frames:
    - Time-based: Accumulate events within fixed time windows
    - Count-based: Accumulate fixed number of events per frame
    - Adaptive: Dynamically adjust accumulation based on event density
    """
    
    def __init__(self, 
                 height: int, 
                 width: int, 
                 accumulation_strategy: str = 'time_based',
                 time_window: float = 50.0,  # milliseconds
                 event_count: int = 1000,
                 num_frames: int = 8,
                 normalize: bool = True,
                 polarity_channels: bool = True):
        """
        Initialize the Event Frame Generator.
        
        Args:
            height: Height of the event frame
            width: Width of the event frame
            accumulation_strategy: 'time_based', 'count_based', or 'adaptive'
            time_window: Time window for time-based accumulation (ms)
            event_count: Number of events for count-based accumulation
            num_frames: Number of frames to generate
            normalize: Whether to normalize the accumulated frames
            polarity_channels: Whether to separate positive/negative events
        """
        self.height = height
        self.width = width
        self.accumulation_strategy = accumulation_strategy
        self.time_window = time_window
        self.event_count = event_count
        self.num_frames = num_frames
        self.normalize = normalize
        self.polarity_channels = polarity_channels
        
        # Number of channels: 2 if polarity_channels else 1
        self.channels = 2 if polarity_channels else 1
    
    def events_to_frames(self, events: Union[event, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Convert events to accumulated frames.
        
        Args:
            events: Event data (can be event object, numpy array, or tensor)
            
        Returns:
            torch.Tensor: Accumulated event frames [C, H, W, T] or [C, H, W]
        """
        if isinstance(events, event):
            return self._event_object_to_frames(events)
        elif isinstance(events, np.ndarray):
            return self._numpy_events_to_frames(events)
        elif isinstance(events, torch.Tensor):
            return self._tensor_events_to_frames(events)
        else:
            raise ValueError(f"Unsupported event type: {type(events)}")
    
    def _event_object_to_frames(self, events: event) -> torch.Tensor:
        """Convert slayerSNN event object to frames."""
        # Extract event data
        x_coords = np.array(events.x, dtype=np.int32)
        y_coords = np.array(events.y, dtype=np.int32)
        polarities = np.array(events.p, dtype=np.int32)
        timestamps = np.array(events.t, dtype=np.float32)
        
        return self._accumulate_events(x_coords, y_coords, polarities, timestamps)
    
    def _numpy_events_to_frames(self, events: np.ndarray) -> torch.Tensor:
        """Convert numpy event array to frames."""
        # Assume format: [timestamp, x, y, polarity]
        timestamps = events[:, 0].astype(np.float32)
        x_coords = events[:, 1].astype(np.int32)
        y_coords = events[:, 2].astype(np.int32)
        polarities = events[:, 3].astype(np.int32)
        
        return self._accumulate_events(x_coords, y_coords, polarities, timestamps)
    
    def _tensor_events_to_frames(self, events: torch.Tensor) -> torch.Tensor:
        """Convert tensor events to frames."""
        if len(events.shape) == 5:  # [B, C, H, W, T] - spike tensor format
            return self._spike_tensor_to_frames(events)
        elif len(events.shape) == 4:  # [C, H, W, T] - spike tensor format
            return self._spike_tensor_to_frames(events.unsqueeze(0)).squeeze(0)
        else:
            # Convert to numpy and process as event list
            events_np = events.cpu().numpy()
            return self._numpy_events_to_frames(events_np)
    
    def _spike_tensor_to_frames(self, spike_tensor: torch.Tensor) -> torch.Tensor:
        """Convert spike tensor to accumulated frames."""
        B, C, H, W, T = spike_tensor.shape
        
        if self.accumulation_strategy == 'time_based':
            return self._time_based_accumulation_from_tensor(spike_tensor)
        elif self.accumulation_strategy == 'count_based':
            return self._count_based_accumulation_from_tensor(spike_tensor)
        else:  # adaptive
            return self._adaptive_accumulation_from_tensor(spike_tensor)
    
    def _accumulate_events(self, x_coords: np.ndarray, y_coords: np.ndarray, 
                          polarities: np.ndarray, timestamps: np.ndarray) -> torch.Tensor:
        """Core accumulation logic for event coordinates."""
        
        if self.accumulation_strategy == 'time_based':
            return self._time_based_accumulation(x_coords, y_coords, polarities, timestamps)
        elif self.accumulation_strategy == 'count_based':
            return self._count_based_accumulation(x_coords, y_coords, polarities, timestamps)
        else:  # adaptive
            return self._adaptive_accumulation(x_coords, y_coords, polarities, timestamps)
    
    def _time_based_accumulation(self, x_coords: np.ndarray, y_coords: np.ndarray,
                                polarities: np.ndarray, timestamps: np.ndarray) -> torch.Tensor:
        """Accumulate events based on fixed time windows."""
        if len(timestamps) == 0:
            return torch.zeros(self.channels, self.height, self.width, self.num_frames)
        
        min_time = timestamps.min()
        max_time = timestamps.max()
        total_time = max_time - min_time
        
        if total_time <= 0:
            # All events at same time
            frame = torch.zeros(self.channels, self.height, self.width)
            for i in range(len(x_coords)):
                if 0 <= x_coords[i] < self.width and 0 <= y_coords[i] < self.height:
                    if self.polarity_channels:
                        channel = 0 if polarities[i] == 0 else 1
                    else:
                        channel = 0
                    frame[channel, y_coords[i], x_coords[i]] += 1
            
            # Repeat for all time frames
            frames = frame.unsqueeze(-1).repeat(1, 1, 1, self.num_frames)
            return frames
        
        frames = torch.zeros(self.channels, self.height, self.width, self.num_frames)
        time_step = total_time / self.num_frames
        
        for i in range(len(x_coords)):
            if 0 <= x_coords[i] < self.width and 0 <= y_coords[i] < self.height:
                # Determine which time frame this event belongs to
                time_idx = int((timestamps[i] - min_time) / time_step)
                time_idx = min(time_idx, self.num_frames - 1)  # Clamp to valid range
                
                if self.polarity_channels:
                    channel = 0 if polarities[i] == 0 else 1
                else:
                    channel = 0
                
                frames[channel, y_coords[i], x_coords[i], time_idx] += 1
        
        if self.normalize:
            frames = self._normalize_frames(frames)
        
        return frames
    
    def _count_based_accumulation(self, x_coords: np.ndarray, y_coords: np.ndarray,
                                 polarities: np.ndarray, timestamps: np.ndarray) -> torch.Tensor:
        """Accumulate events based on fixed event counts."""
        frames = torch.zeros(self.channels, self.height, self.width, self.num_frames)
        
        if len(x_coords) == 0:
            return frames
        
        events_per_frame = max(1, len(x_coords) // self.num_frames)
        
        for frame_idx in range(self.num_frames):
            start_idx = frame_idx * events_per_frame
            end_idx = min((frame_idx + 1) * events_per_frame, len(x_coords))
            
            for i in range(start_idx, end_idx):
                if 0 <= x_coords[i] < self.width and 0 <= y_coords[i] < self.height:
                    if self.polarity_channels:
                        channel = 0 if polarities[i] == 0 else 1
                    else:
                        channel = 0
                    
                    frames[channel, y_coords[i], x_coords[i], frame_idx] += 1
        
        if self.normalize:
            frames = self._normalize_frames(frames)
        
        return frames
    
    def _adaptive_accumulation(self, x_coords: np.ndarray, y_coords: np.ndarray,
                              polarities: np.ndarray, timestamps: np.ndarray) -> torch.Tensor:
        """Adaptive accumulation based on event density."""
        # For now, use time-based as fallback
        # TODO: Implement adaptive strategy based on local event density
        return self._time_based_accumulation(x_coords, y_coords, polarities, timestamps)
    
    def _time_based_accumulation_from_tensor(self, spike_tensor: torch.Tensor) -> torch.Tensor:
        """Time-based accumulation from spike tensor."""
        B, C, H, W, T = spike_tensor.shape
        
        # Calculate how many time steps to group together
        time_step = max(1, T // self.num_frames)
        
        frames = torch.zeros(B, self.channels, H, W, self.num_frames, device=spike_tensor.device)
        
        for frame_idx in range(self.num_frames):
            start_t = frame_idx * time_step
            end_t = min((frame_idx + 1) * time_step, T)
            
            # Sum over the time window
            accumulated = spike_tensor[:, :, :, :, start_t:end_t].sum(dim=-1)
            
            if self.polarity_channels and C == 2:
                frames[:, :, :, :, frame_idx] = accumulated
            else:
                # Sum across channels if not using polarity channels
                frames[:, 0, :, :, frame_idx] = accumulated.sum(dim=1)
        
        if self.normalize:
            frames = self._normalize_frames(frames)
        
        return frames.squeeze(0) if B == 1 else frames
    
    def _count_based_accumulation_from_tensor(self, spike_tensor: torch.Tensor) -> torch.Tensor:
        """Count-based accumulation from spike tensor."""
        # For spike tensors, time-based makes more sense
        return self._time_based_accumulation_from_tensor(spike_tensor)
    
    def _adaptive_accumulation_from_tensor(self, spike_tensor: torch.Tensor) -> torch.Tensor:
        """Adaptive accumulation from spike tensor."""
        return self._time_based_accumulation_from_tensor(spike_tensor)
    
    def _normalize_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Normalize accumulated frames."""
        # Normalize each frame independently
        if len(frames.shape) == 4:  # [C, H, W, T]
            for t in range(frames.shape[-1]):
                frame = frames[:, :, :, t]
                max_val = frame.max()
                if max_val > 0:
                    frames[:, :, :, t] = frame / max_val
        elif len(frames.shape) == 5:  # [B, C, H, W, T]
            for b in range(frames.shape[0]):
                for t in range(frames.shape[-1]):
                    frame = frames[b, :, :, :, t]
                    max_val = frame.max()
                    if max_val > 0:
                        frames[b, :, :, :, t] = frame / max_val
        
        return frames


def create_event_frames_from_spike_tensor(spike_tensor: torch.Tensor, 
                                        num_frames: int = 8,
                                        strategy: str = 'time_based') -> torch.Tensor:
    """
    Convenience function to create event frames from spike tensor.
    
    Args:
        spike_tensor: Input spike tensor [B, C, H, W, T] or [C, H, W, T]
        num_frames: Number of output frames
        strategy: Accumulation strategy
        
    Returns:
        torch.Tensor: Event frames [B, C, H, W, num_frames] or [C, H, W, num_frames]
    """
    if len(spike_tensor.shape) == 4:
        C, H, W, T = spike_tensor.shape
        spike_tensor = spike_tensor.unsqueeze(0)  # Add batch dimension
        squeeze_batch = True
    else:
        B, C, H, W, T = spike_tensor.shape
        squeeze_batch = False
    
    generator = EventFrameGenerator(
        height=H, 
        width=W, 
        accumulation_strategy=strategy,
        num_frames=num_frames,
        polarity_channels=(C == 2)
    )
    
    frames = generator.events_to_frames(spike_tensor)
    
    if squeeze_batch and len(frames.shape) == 5:
        frames = frames.squeeze(0)
    
    return frames
