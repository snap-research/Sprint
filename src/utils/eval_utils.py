from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
import re
import os
import scipy
from safetensors.torch import load_file, save_file
import glob
import imageio
import random


def calculate_inception_stats_cifar(arr, detector_net=None, detector_kwargs=None, batch_size=100, device='cpu'):
    num_samples = arr.shape[0]
    count = 0
    mu = torch.zeros([2048], dtype=torch.float64, device=device)
    sigma = torch.zeros([2048, 2048], dtype=torch.float64, device=device)

    for k in range((arr.shape[0] - 1) // batch_size + 1):
        mic_img = arr[k * batch_size: (k + 1) * batch_size]
        mic_img = torch.tensor(mic_img).permute(0, 3, 1, 2).to(device)
        features = detector_net(mic_img, **detector_kwargs).to(torch.float64)
        if count + mic_img.shape[0] > num_samples:
            remaining_num_samples = num_samples - count
        else:
            remaining_num_samples = mic_img.shape[0]
        mu += features[:remaining_num_samples].sum(0)
        sigma += features[:remaining_num_samples].T @ features[:remaining_num_samples]
        count = count + remaining_num_samples
    assert count == num_samples
    mu /= num_samples
    sigma -= mu.ger(mu) * num_samples
    sigma /= num_samples - 1
    mu = mu.cpu().numpy()
    sigma = sigma.cpu().numpy()
    return mu, sigma

def calculate_inception_stats_imagenet(arr, evaluator, batch_size=100, device='cpu'):
    print("computing sample batch activations...")
    sample_acts = evaluator.read_activations(arr)
    print("computing/reading sample batch statistics...")
    sample_stats, sample_stats_spatial = tuple(evaluator.compute_statistics(x) for x in sample_acts)
    return sample_acts, sample_stats, sample_stats_spatial
    

def compute_fid(mu, sigma, ref_mu=None, ref_sigma=None):
    m = np.square(mu - ref_mu).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, ref_sigma), disp=False)
    fid = m + np.trace(sigma + ref_sigma - s * 2)
    fid = float(np.real(fid))
    return fid


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

def create_npz_from_video_folder(video_dir, frames_per_video=16, size=None, num=None, recursive=True):
    """
    Builds a single .npz file from a folder of .mp4 videos.

    Args:
        video_dir: Directory containing .mp4 videos (can be nested if recursive=True)
        frames_per_video: Number of frames to sample uniformly from each video
        size: If provided, resize frames to (height, width). Accepts int (square) or (H, W) tuple
        num: Maximum number of videos to process (None for all)
        recursive: Search subdirectories for .mp4 files

    Returns:
        npz_path: Path to the saved .npz file with key 'videos' of shape [N, T, H, W, 3]
    """
    if not os.path.isdir(video_dir):
        raise ValueError(f"Not a directory: {video_dir}")

    pattern = os.path.join(video_dir, '**', '*.mp4') if recursive else os.path.join(video_dir, '*.mp4')
    video_paths = sorted(glob.glob(pattern, recursive=recursive))
    if len(video_paths) == 0:
        raise ValueError(f"No .mp4 files found in {video_dir}")

    if num is not None:
        video_paths = video_paths[:num]

    def _maybe_resize(frame, size):
        if size is None:
            return frame
        if isinstance(size, int):
            target_size = (size, size)
        else:
            target_size = (int(size[0]), int(size[1]))
        # PIL expects (W, H)
        pil_img = Image.fromarray(frame)
        pil_img = pil_img.resize((target_size[1], target_size[0]), Image.BICUBIC)
        return np.asarray(pil_img)

    videos = []
    failed = 0
    for vp in tqdm(video_paths, desc="Building .npz file from videos"):
        try:
            reader = imageio.get_reader(vp, format='ffmpeg')
            meta = reader.get_meta_data()
            total_frames = meta.get('nframes', None)
            # Some containers may not report nframes reliably; fall back by iterating once if needed
            if total_frames is None or total_frames <= 0 or total_frames == float('inf'):
                # Probe frames to count
                temp_frames = []
                for fr in reader:
                    temp_frames.append(fr)
                total_frames = len(temp_frames)
                reader.close()
                if total_frames == 0:
                    raise ValueError("No frames decoded")
                # Re-open for indexed access below
                reader = imageio.get_reader(vp, format='ffmpeg')
            # Choose indices uniformly across the video
            if total_frames < frames_per_video:
                indices = np.linspace(0, total_frames - 1, total_frames, dtype=int)
            else:
                indices = np.linspace(0, total_frames - 1, frames_per_video, dtype=int)

            sampled = []
            for idx in indices:
                frame = reader.get_data(int(idx))
                if frame.ndim == 2:
                    # Grayscale -> RGB
                    frame = np.stack([frame, frame, frame], axis=-1)
                frame = _maybe_resize(frame, size)
                # Ensure uint8
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                sampled.append(frame)
            reader.close()

            if len(sampled) == 0:
                raise ValueError("No frames sampled")

            # Stack to [T, H, W, C]
            video_arr = np.stack(sampled, axis=0)
            videos.append(video_arr)
        except Exception as e:
            failed += 1
            print(f"Skipping {vp}: {e}")
            continue

    if len(videos) == 0:
        raise ValueError("No valid videos could be processed")

    # To build a consistent tensor, ensure all videos have same [T,H,W].
    # Pad or trim temporally to frames_per_video, and resize ensured earlier.
    # Determine target T
    target_T = frames_per_video
    # Determine target H,W from first video
    H, W = videos[0].shape[1], videos[0].shape[2]
    processed = []
    for v in videos:
        # Temporal adjust: pad or trim
        if v.shape[0] < target_T:
            pad_count = target_T - v.shape[0]
            pads = np.repeat(v[-1:,:,:,:], pad_count, axis=0)
            v = np.concatenate([v, pads], axis=0)
        elif v.shape[0] > target_T:
            v = v[:target_T]
        # Spatial adjust if needed (in case a file differed in size after decoding)
        if v.shape[1] != H or v.shape[2] != W:
            v_resized = []
            for fr in v:
                fr = Image.fromarray(fr).resize((W, H), Image.BICUBIC)
                v_resized.append(np.asarray(fr))
            v = np.stack(v_resized, axis=0)
        processed.append(v)

    videos_np = np.stack(processed, axis=0)  # [N, T, H, W, C]
    npz_path = f"{video_dir}/{video_dir}.npz"
    np.savez(npz_path, videos=videos_np)
    print(f"Saved .npz file to {npz_path} [shape={videos_np.shape}].")
    if failed > 0:
        print(f"Warning: Failed to process {failed} videos")
    return npz_path

def load_images_from_npz(npz_path, max_samples=None):
    """
    Load images from NPZ file
    
    Args:
        npz_path: Path to NPZ file containing images
        max_samples: Maximum number of samples to load (None for all)
    
    Returns:
        images: numpy array of shape [N, H, W, 3] with values in [0, 255]
    """
    print(f"Loading images from NPZ: {npz_path}")
    data = np.load(npz_path)
    
    # Handle different NPZ formats
    if 'arr_0' in data:
        images = data['arr_0']
    elif 'images' in data:
        images = data['images']
    else:
        # Take the first array found
        key = list(data.keys())[0]
        images = data[key]
        print(f"Using key '{key}' from NPZ file")
    
    print(f"Original NPZ shape: {images.shape}")
    print(f"Original value range: [{images.min():.3f}, {images.max():.3f}]")
    
    # Ensure images are in correct format [N, H, W, 3] with values in [0, 255]
    if images.dtype != np.uint8:
        if images.max() <= 1.0:
            # Images are in [0, 1] range, convert to [0, 255]
            images = (images * 255).astype(np.uint8)
            print("Converted from [0, 1] to [0, 255] range")
        elif images.min() >= -1.0 and images.max() <= 1.0:
            # Images are in [-1, 1] range, convert to [0, 255]
            images = ((images + 1.0) * 127.5).astype(np.uint8)
            print("Converted from [-1, 1] to [0, 255] range")
        else:
            # Assume images are already in [0, 255] range
            images = images.astype(np.uint8)
    
    # Handle different channel arrangements
    if len(images.shape) == 4:
        # Check if we need to transpose from NCHW to NHWC
        if images.shape[1] == 3 and images.shape[3] != 3:
            # Likely NCHW format, convert to NHWC
            images = np.transpose(images, (0, 2, 3, 1))
            print("Converted from NCHW to NHWC format")
        elif images.shape[3] == 3:
            # Already in NHWC format
            pass
        else:
            raise ValueError(f"Unexpected image shape: {images.shape}. Expected format: [N, H, W, 3] or [N, 3, H, W]")
    else:
        raise ValueError(f"Unexpected image array dimensions: {images.shape}. Expected 4D array.")
    
    if max_samples and images.shape[0] > max_samples:
        print(f"Limiting to {max_samples} samples from {images.shape[0]} total")
        images = images[:max_samples]
    
    print(f"Final image shape: {images.shape}")
    print(f"Final value range: [{images.min()}, {images.max()}]")
    
    return images

def load_videos_from_npz(npz_path, max_samples=None):
    """
    Load videos from NPZ file
    
    Args:
        npz_path: Path to NPZ file containing videos
        max_samples: Maximum number of samples to load (None for all)
    """
    data = np.load(npz_path, allow_pickle=True)
    videos = data['videos']
    import pdb; pdb.set_trace()
    return videos

def load_images_from_folder(folder_path, max_samples=None, image_extensions=None):
    """
    Load images from a folder
    
    Args:
        folder_path: Path to folder containing images
        max_samples: Maximum number of samples to load (None for all)
        image_extensions: List of image extensions to look for
    
    Returns:
        images: numpy array of shape [N, H, W, 3] with values in [0, 255]
    """
    if image_extensions is None:
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.webp']
    
    print(f"Loading images from: {folder_path}")
    
    # Find all image files
    image_paths = []
    for ext in image_extensions:
        pattern = os.path.join(folder_path, '**', ext)
        image_paths.extend(glob.glob(pattern, recursive=True))
        # Also try uppercase extensions
        pattern = os.path.join(folder_path, '**', ext.upper())
        image_paths.extend(glob.glob(pattern, recursive=True))
    
    # Remove duplicates and sort
    image_paths = sorted(list(set(image_paths)))
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {folder_path} with extensions {image_extensions}")
    
    print(f"Found {len(image_paths)} images")
    
    if max_samples and len(image_paths) > max_samples:
        print(f"Limiting to {max_samples} samples")
        image_paths = image_paths[:max_samples]
    
    # Load images
    images = []
    failed_count = 0
    
    for img_path in tqdm(image_paths, desc="Loading images"):
        try:
            # Load image and convert to RGB
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img)
            
            # Ensure the image has the right shape
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                images.append(img_array)
            else:
                print(f"Skipping {img_path}: unexpected shape {img_array.shape}")
                failed_count += 1
                
        except Exception as e:
            print(f"Failed to load {img_path}: {e}")
            failed_count += 1
            continue
    
    if len(images) == 0:
        raise ValueError("No valid images could be loaded")
    
    if failed_count > 0:
        print(f"Warning: Failed to load {failed_count} images")
    
    # Convert to numpy array
    images = np.stack(images, axis=0)
    print(f"Successfully loaded {images.shape[0]} images with shape {images.shape}")
    print(f"Image value range: [{images.min()}, {images.max()}]")
    
    return images

def load_videos_from_folder(folder_path, max_samples=None, video_extensions=None):
    """
    Load videos from a folder
    
    Args:
        folder_path: Path to folder containing videos
        max_samples: Maximum number of samples to load (None for all)
        video_extensions: List of video extensions to look for
    """

    if video_extensions is None:
        video_extensions = ['*.mp4']
    
    print(f"Loading videos from: {folder_path}")
    
    # Find all video files
    video_paths = []
    for ext in video_extensions:
        pattern = os.path.join(folder_path, '**', ext)
        video_paths.extend(glob.glob(pattern, recursive=True))
        # Also try uppercase extensions
        pattern = os.path.join(folder_path, '**', ext.upper())
        video_paths.extend(glob.glob(pattern, recursive=True))
    
    # Remove duplicates and sort
    video_paths = sorted(list(set(video_paths)))
    
    if len(video_paths) == 0:
        raise ValueError(f"No videos found in {folder_path} with extensions {video_extensions}")
    
    print(f"Found {len(video_paths)} videos")
    
    if max_samples and len(video_paths) > max_samples:
        print(f"Limiting to {max_samples} samples")
        video_paths = video_paths[:max_samples]
    
    # Load videos
    videos = []
    failed_count = 0
    
    for vid_path in tqdm(video_paths, desc="Loading videos"):
        try:
            reader = imageio.get_reader(vid_path, format='ffmpeg')
            frames = []
            for frame in reader:
                # Ensure RGB
                if frame.ndim == 2:
                    frame = np.stack([frame, frame, frame], axis=-1)
                frame = np.array(frame)
                frames.append(frame)
            reader.close()
            
            if len(frames) == 0:
                print(f"Skipping {vid_path}: no frames decoded")
                failed_count += 1
                continue
            
            video_arr = np.stack(frames, axis=0)  # [T, H, W, C]
            
            # Ensure the video has the right shape
            if len(video_arr.shape) == 4 and video_arr.shape[3] == 3:
                videos.append(video_arr)
            else:
                print(f"Skipping {vid_path}: unexpected shape {video_arr.shape}")
                failed_count += 1
        except Exception as e:
            print(f"Failed to load {vid_path}: {e}")
            failed_count += 1
            continue
    
    if len(videos) == 0:
        raise ValueError("No valid videos could be loaded")
    
    if failed_count > 0:
        print(f"Warning: Failed to load {failed_count} videos")
    
    # Convert to numpy array
    videos = np.stack(videos, axis=0)  # [N, T, H, W, 3]
    print(f"Successfully loaded {videos.shape[0]} videos with shape {videos.shape}")
    print(f"Video value range: [{videos.min()}, {videos.max()}]")
    
    return videos

def iter_sampled_frames_from_videos(
    folder_path,
    frames_per_video: int = 1,
    resize: tuple[int, int] | int | None = None,
    max_videos: int | None = None,
    video_extensions=None,
):
    """
    Iterate over sampled frames from videos in a folder (single pass per video).

    - Uses reservoir sampling to select `frames_per_video` frames uniformly per video
      without knowing the total number of frames in advance.
    - Each yielded frame is a numpy array [H, W, 3] in uint8.

    Args:
        folder_path: Path to folder containing videos
        frames_per_video: Number of frames to sample per video (uniform, single pass)
        resize: If provided, resize frames to (H, W) or int for square; else keep original
        max_videos: Process up to this many videos (None for all)
        video_extensions: List of glob patterns for video extensions (default ['*.mp4'])
    Yields:
        np.ndarray: Frame array [H, W, 3] uint8
    """

    if video_extensions is None:
        video_extensions = ['*.mp4']

    # Find all video files
    video_paths = []
    for ext in video_extensions:
        pattern = os.path.join(folder_path, '**', ext)
        video_paths.extend(glob.glob(pattern, recursive=True))
        pattern = os.path.join(folder_path, '**', ext.upper())
        video_paths.extend(glob.glob(pattern, recursive=True))

    video_paths = sorted(list(set(video_paths)))

    if len(video_paths) == 0:
        raise ValueError(f"No videos found in {folder_path} with extensions {video_extensions}")

    if max_videos is not None and len(video_paths) > max_videos:
        video_paths = video_paths[:max_videos]

    def _maybe_resize(frame):
        if resize is None:
            return frame
        if isinstance(resize, int):
            target_h, target_w = resize, resize
        else:
            target_h, target_w = int(resize[0]), int(resize[1])
        pil_img = Image.fromarray(frame)
        pil_img = pil_img.resize((target_w, target_h), Image.BICUBIC)
        return np.asarray(pil_img)

    for vid_path in tqdm(video_paths, desc="Iterating videos"):
        try:
            reader = imageio.get_reader(vid_path, format='ffmpeg')
        except Exception as e:
            print(f"Failed to open {vid_path}: {e}")
            continue

        reservoir = []
        seen = 0
        try:
            for frame in reader:
                if frame.ndim == 2:
                    frame = np.stack([frame, frame, frame], axis=-1)
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                # Reservoir sampling (k = frames_per_video)
                if seen < frames_per_video:
                    reservoir.append(frame)
                else:
                    j = random.randint(0, seen)
                    if j < frames_per_video:
                        reservoir[j] = frame
                seen += 1
        except Exception as e:
            print(f"Failed while reading {vid_path}: {e}")
        finally:
            try:
                reader.close()
            except Exception:
                pass

        if seen == 0:
            print(f"Skipping {vid_path}: no frames decoded")
            continue

        # Yield selected frames (optional resize)
        for fr in reservoir:
            fr_resized = _maybe_resize(fr)
            yield fr_resized

def iter_sampled_from_images(
    folder_path,
    resize: tuple[int, int] | int | None = None,
    max_images: int | None = None,
    image_extensions=None,
):
    """
    Iterate over sampled images from a folder.

    - Uses reservoir sampling to select `max_images` images uniformly.
    - Each yielded image is a numpy array [H, W, 3] in uint8.

    Args:
        folder_path: Path to folder containing images
        resize: If provided, resize images to (H, W) or int for square; else keep original
        max_images: Process up to this many images (None for all)
        image_extensions: List of glob patterns for image extensions (default ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.webp'])
    Yields:
        np.ndarray: Image array [H, W, 3] uint8
    """

    if image_extensions is None:
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.webp']

    # Find all image files
    image_paths = []
    for ext in image_extensions:
        pattern = os.path.join(folder_path, '**', ext)
        image_paths.extend(glob.glob(pattern, recursive=True))
        pattern = os.path.join(folder_path, '**', ext.upper())
        image_paths.extend(glob.glob(pattern, recursive=True))

    image_paths = sorted(list(set(image_paths)))

    if len(image_paths) == 0:
        raise ValueError(f"No images found in {folder_path} with extensions {image_extensions}")

    if max_images is not None and len(image_paths) > max_images:
        image_paths = image_paths[:max_images]

    def _maybe_resize(image):
        if resize is None:
            return image
        if isinstance(resize, int):
            target_h, target_w = resize, resize
        else:
            target_h, target_w = int(resize[0]), int(resize[1])
        pil_img = Image.fromarray(image)
        pil_img = pil_img.resize((target_w, target_h), Image.BICUBIC)
        return np.asarray(pil_img)

    # If max_images is None, stream-yield every image; else use reservoir sampling
    if max_images is None:
        for image_path in tqdm(image_paths, desc="Iterating images"):
            try:
                with Image.open(image_path) as img:
                    img = img.convert('RGB')
                    arr = np.asarray(img)
            except Exception as e:
                print(f"Failed to open {image_path}: {e}")
                continue

            if arr.dtype != np.uint8:
                arr = arr.astype(np.uint8)

            arr = _maybe_resize(arr)
            yield arr
        return

    # Reservoir sampling to select max_images uniformly from the directory
    reservoir = []
    seen = 0
    for image_path in tqdm(image_paths, desc="Iterating images"):
        try:
            with Image.open(image_path) as img:
                img = img.convert('RGB')
                arr = np.asarray(img)
        except Exception as e:
            print(f"Failed to open {image_path}: {e}")
            continue

        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)

        if seen < max_images:
            reservoir.append(arr)
        else:
            j = random.randint(0, seen)
            if j < max_images:
                reservoir[j] = arr
        seen += 1

    if seen == 0:
        raise ValueError(f"No valid images could be read in {folder_path}")

    # Yield selected images (optional resize)
    for im in reservoir:
        im_resized = _maybe_resize(im)
        yield im_resized

def iter_sampled_from_images_to_video(
    folder_path,
    frames_per_video: int | None = None,
    resize: tuple[int, int] | int | None = None,
    batch_size: int = 1,
    max_videos: int | None = None,
    image_extensions=None,
):
    """
    Stream videos constructed from image frames on disk and yield them in batches.

    Expects filenames of the form "<video_id>-<frame_number>.png" (e.g., "sampleA-000012.png").
    Frames are grouped by <video_id> and ordered by <frame_number>. This maintains
    streaming behavior by only holding one video's frames (and the current batch)
    in memory at a time.

    Args:
        folder_path: Root directory containing frame images.
        frames_per_video: If provided, trim or pad each video to this many frames
            (uniform subsample when trimming, last-frame repeat when padding).
        resize: If provided, resize frames to (H, W) or an int for square.
        batch_size: Number of videos per batch to yield.
        max_videos: Process up to this many videos (None for all).
        image_extensions: Glob patterns for image files (default includes png/jpg/...); for
            best performance with your setup, you can pass ['*.png'].

    Yields:
        np.ndarray with shape [B, T, H, W, 3] (uint8), where B <= batch_size.
        If frames_per_video is None, T equals the number of frames in each video; in that
        case videos in the same batch must share the same T,H,W to stack. If shapes differ,
        the function flushes the current batch and starts a new one so streaming continues.
    """

    if image_extensions is None:
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.webp']

    # Discover image files
    image_paths = []
    for ext in image_extensions:
        pattern = os.path.join(folder_path, '**', ext)
        image_paths.extend(glob.glob(pattern, recursive=True))
        pattern = os.path.join(folder_path, '**', ext.upper())
        image_paths.extend(glob.glob(pattern, recursive=True))
    image_paths = sorted(list(set(image_paths)))

    if len(image_paths) == 0:
        raise ValueError(f"No images found in {folder_path} with extensions {image_extensions}")

    # Parse video_id and frame_number from filenames; group by video_id
    def _parse_video_and_frame(path):
        base = os.path.splitext(os.path.basename(path))[0]
        m = re.match(r'^(.+)-(\d+)$', base)
        if m:
            raw_video_id, frame_str = m.group(1), m.group(2)
            try:
                frame_idx = int(frame_str)
            except Exception:
                frame_idx = 0
        else:
            raw_video_id, frame_idx = base, 0
        # Include subdirectory in the key to avoid collisions
        rel_dir = os.path.relpath(os.path.dirname(path), folder_path)
        video_key = os.path.join(rel_dir, raw_video_id) if rel_dir != '.' else raw_video_id
        return video_key, frame_idx

    records = []
    for p in image_paths:
        vid_key, frame_idx = _parse_video_and_frame(p)
        records.append((vid_key, frame_idx, p))
    # Sort by video then frame index
    records.sort(key=lambda x: (x[0], x[1]))
    #import pdb; pdb.set_trace()

    def _maybe_resize(image):
        if resize is None:
            return image
        if isinstance(resize, int):
            target_h, target_w = resize, resize
        else:
            target_h, target_w = int(resize[0]), int(resize[1])
        pil_img = Image.fromarray(image)
        pil_img = pil_img.resize((target_w, target_h), Image.BICUBIC)
        return np.asarray(pil_img)

    def _finalize_video(frames_list):
        if len(frames_list) == 0:
            return None
        # Optionally trim/subsample or pad to frames_per_video
        if frames_per_video is not None:
            if len(frames_list) < frames_per_video:
                pad_count = frames_per_video - len(frames_list)
                frames_list = frames_list + [frames_list[-1]] * pad_count
            elif len(frames_list) > frames_per_video:
                idxs = np.linspace(0, len(frames_list) - 1, frames_per_video, dtype=int)
                frames_list = [frames_list[i] for i in idxs]
        # Resize and stack
        processed = []
        for fr in frames_list:
            if fr.dtype != np.uint8:
                fr = fr.astype(np.uint8)
            # fr = _maybe_resize(fr)
            processed.append(fr)
        try:
            video_np = np.stack(processed, axis=0)  # [T, H, W, 3]
        except Exception:
            # Fallback: enforce same spatial size via PIL if any mismatch slipped through
            H, W = processed[0].shape[0], processed[0].shape[1]
            fixed = []
            for fr in processed:
                if fr.shape[0] != H or fr.shape[1] != W:
                    fr = np.asarray(Image.fromarray(fr).resize((W, H), Image.BICUBIC))
                fixed.append(fr.astype(np.uint8))
            video_np = np.stack(fixed, axis=0)
        return video_np

    batch = []
    videos_emitted = 0
    current_key = None
    current_frames = []

    def _maybe_yield_batch(force=False):
        nonlocal batch, videos_emitted
        if len(batch) == 0:
            return
        if force or len(batch) >= batch_size:
            try:
                batch_np = np.stack(batch, axis=0)  # [B, T, H, W, 3]
                yield_obj = batch_np
            except Exception:
                # If shapes differ (e.g., variable T/H/W), flush one by one
                for v in batch:
                    yield v[np.newaxis, ...]
                batch = []
                return
            batch = []
            yield yield_obj

    # Iterate sorted records and build videos on the fly
    for vid_key, frame_idx, path in tqdm(records, desc="Grouping frames into videos"):
        if current_key is None:
            current_key = vid_key
        if vid_key != current_key:
            # finalize previous video
            video_np = _finalize_video(current_frames)
            if video_np is not None:
                batch.append(video_np)
                videos_emitted += 1
                # Yield batch when full
                for out in _maybe_yield_batch(force=False):
                    yield out
                if max_videos is not None and videos_emitted >= max_videos:
                    # Flush any remaining partial batch before exit
                    for out in _maybe_yield_batch(force=True):
                        yield out
                    return
            # reset for new video
            current_key = vid_key
            current_frames = []

        # load frame and accumulate
        try:
            with Image.open(path) as img:
                img = img.convert('RGB')
                arr = np.asarray(img)
        except Exception as e:
            print(f"Failed to open {path}: {e}")
            continue
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        current_frames.append(arr)

    # finalize last video
    video_np = _finalize_video(current_frames)
    if video_np is not None:
        batch.append(video_np)
        videos_emitted += 1

    # Yield any remaining batch
    for out in _maybe_yield_batch(force=True):
        yield out

def load_videos_from_folder_as_images(
    folder_path,
    frames_per_video: int | None = None,
    resize: tuple[int, int] | int | None = None,
    max_videos: int | None = None,
    video_extensions=None,
):
    """
    Load videos and return flattened frames as [N*T, H, W, 3].

    - Reads each video once. If frames_per_video is provided, uniformly samples that
      many frames per video; otherwise decodes all frames.
    - Optionally resizes frames.

    Args:
        folder_path: Root directory containing videos
        frames_per_video: Number of frames to sample uniformly per video (None = all)
        resize: If provided, resize frames to (H, W) or int for square
        max_videos: Limit number of videos processed
        video_extensions: Glob patterns to match video files
    Returns:
        np.ndarray shaped [N*T, H, W, 3] (uint8)
    """

    if video_extensions is None:
        video_extensions = ['*.mp4']

    # Build file list
    video_paths = []
    for ext in video_extensions:
        pattern = os.path.join(folder_path, '**', ext)
        video_paths.extend(glob.glob(pattern, recursive=True))
        pattern = os.path.join(folder_path, '**', ext.upper())
        video_paths.extend(glob.glob(pattern, recursive=True))
    video_paths = sorted(list(set(video_paths)))

    if len(video_paths) == 0:
        raise ValueError(f"No videos found in {folder_path} with extensions {video_extensions}")

    if max_videos is not None and len(video_paths) > max_videos:
        video_paths = video_paths[:max_videos]

    def _maybe_resize(frame):
        if resize is None:
            return frame
        if isinstance(resize, int):
            target_h, target_w = resize, resize
        else:
            target_h, target_w = int(resize[0]), int(resize[1])
        pil_img = Image.fromarray(frame)
        pil_img = pil_img.resize((target_w, target_h), Image.BICUBIC)
        return np.asarray(pil_img)

    flat_frames = []
    for vid_path in tqdm(video_paths, desc="Flattening videos to frames"):
        try:
            reader = imageio.get_reader(vid_path, format='ffmpeg')
        except Exception as e:
            print(f"Failed to open {vid_path}: {e}")
            continue

        try:
            meta = reader.get_meta_data()
            total_frames = meta.get('nframes', None)
            # fallback if missing
            if total_frames is None or total_frames <= 0 or total_frames == float('inf'):
                total_frames = 0
                for _ in reader:
                    total_frames += 1
                reader.close()
                reader = imageio.get_reader(vid_path, format='ffmpeg')

            if frames_per_video is None or frames_per_video >= total_frames:
                indices = range(total_frames)
            else:
                indices = np.linspace(0, total_frames - 1, frames_per_video, dtype=int)

            for idx in indices:
                try:
                    frame = reader.get_data(int(idx))
                except Exception:
                    # fallback to sequential read if random access fails
                    frame = None
                if frame is None:
                    continue
                if frame.ndim == 2:
                    frame = np.stack([frame, frame, frame], axis=-1)
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                frame = _maybe_resize(frame)
                flat_frames.append(frame)
        except Exception as e:
            print(f"Failed while reading {vid_path}: {e}")
        finally:
            try:
                reader.close()
            except Exception:
                pass

    if len(flat_frames) == 0:
        raise ValueError("No frames decoded from any video")

    return np.stack(flat_frames, axis=0)