# Personalised Multimodal Soccer Video Summarisation

This repository contains the implementation for a dissertation project on personalised soccer video summarisation using SoccerNet videos and multimodal scoring. The system takes a SoccerNet match half and its `Labels-v2.json` annotation file, generates candidate clips around annotated events, scores those clips using selected modalities, and exports a short highlight summary video. 

## Main Features

- Parses SoccerNet `Labels-v2.json` annotation files.
- Generates candidate highlight clips around annotated soccer events.
- Supports different user preference profiles:
  - `goals`
  - `defense`
  - `discipline`
  - `balanced`
- Scores candidate clips using a weighted fusion of:
  - event annotation relevance
  - CLIP-based scene recognition
  - Grounding DINO object grounding
  - audio excitement using RMS energy and onset strength
  - dense optical-flow motion intensity
- Uses MMR-style selection to reduce repeated or overlapping clips.
- Exports a playable summary video using `ffmpeg`.
- Saves a JSON report containing selected clips, scores, and evaluation metrics.

## Repository Structure

```text
tnn33Dissertation/
├── README.md                # Project documentation
├── download_soccernet.py    # Helper script for downloading SoccerNet validation data
└── final_soccer.py          # Main multimodal summarisation pipeline
```

Expected local data structure after downloading SoccerNet data:

```text
data/
└── SoccerNet/
    └── <league>/<season>/<match>/
        ├── Labels-v2.json
        ├── 1_224p.mkv
        └── 2_224p.mkv
```

The SoccerNet dataset is not included in this repository because access requires permission from SoccerNet.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/tiennamnguyen/tnn33Dissertation.git
cd tnn33Dissertation
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```


### 3. Install Python dependencies

```bash
pip install --upgrade pip
pip install numpy opencv-python pillow tqdm librosa SoccerNet transformers torch torchvision
```

The heavier dependencies are only required when using the optional multimodal features:

- `transformers` and `torch` are required for CLIP and Grounding DINO.
- `librosa` is required for audio scoring.
- `opencv-python` is required for video reading and optical flow.
- `SoccerNet` is required for the download helper script.

### 4. Install ffmpeg

`ffmpeg` is required for audio extraction and for exporting the final summary video.

On macOS using Homebrew:

```bash
brew install ffmpeg
```

On Ubuntu/Linux:

```bash
sudo apt update
sudo apt install ffmpeg
```

Check that it installed correctly:

```bash
ffmpeg -version
ffprobe -version
```

## Downloading SoccerNet Data

Before running the main pipeline, obtain access to the SoccerNet dataset. Once access is granted, edit `download_soccernet.py` and replace:

```python
mySoccerNetDownloader.password = "REQUIRES_PERMISSION"
```

with your SoccerNet download password.

Then run:

```bash
python download_soccernet.py
```

The script downloads the validation split annotations and 224p match-half videos into:

```text
data/SoccerNet/
```

## Running the Summarisation Pipeline

The main script is `final_soccer.py`.

Basic annotation-only run:

```bash
python final_soccer.py \
  --video "data/SoccerNet/path/to/match/1_224p.mkv" \
  --labels "data/SoccerNet/path/to/match/Labels-v2.json" \
  --half 1 \
  --profile balanced \
  --output summary_v2.mp4 \
  --report summary_report_v2.json
```

Full multimodal run:

```bash
python final_soccer.py \
  --video "data/SoccerNet/path/to/match/1_224p.mkv" \
  --labels "data/SoccerNet/path/to/match/Labels-v2.json" \
  --half 1 \
  --profile goals \
  --use_clip \
  --use_grounding \
  --use_audio \
  --use_flow \
  --output highlights_v2.mp4 \
  --report highlights_report_v2.json
```

## Command-Line Arguments

| Argument | Description |
|---|---|
| `--video` | Path to the SoccerNet video half, for example `1_224p.mkv` or `2_224p.mkv`. |
| `--labels` | Path to the corresponding `Labels-v2.json` file. |
| `--half` | Match half to process. Use `1` or `2`. |
| `--profile` | User preference profile: `goals`, `defense`, `discipline`, or `balanced`. |
| `--output` | Output path for the generated summary video. |
| `--report` | Output path for the JSON report. |
| `--pre_context` | Seconds included before each event timestamp. Default: `8.0`. |
| `--post_context` | Seconds included after each event timestamp. Default: `10.0`. |
| `--max_summary_seconds` | Maximum total summary duration. Default: `120.0`. |
| `--top_k` | Maximum number of clips selected. Default: `12`. |
| `--min_gap` | Minimum time gap between selected clips. Default: `5.0`. |
| `--lambda_mmr` | Relevance-diversity trade-off used during MMR selection. Default: `0.70`. |
| `--use_clip` | Enables CLIP scene recognition. |
| `--use_grounding` | Enables Grounding DINO object grounding. |
| `--use_audio` | Enables audio excitement scoring. |
| `--use_flow` | Enables dense optical-flow motion scoring. |
| `--clip_frames` | Number of frames sampled per candidate clip for visual scoring. Default: `8`. |
| `--w_event` | Fusion weight for annotation/event score. Default: `0.45`. |
| `--w_scene` | Fusion weight for CLIP scene score. Default: `0.20`. |
| `--w_grounding` | Fusion weight for grounding score. Default: `0.10`. |
| `--w_audio` | Fusion weight for audio score. Default: `0.15`. |
| `--w_flow` | Fusion weight for optical-flow score. Default: `0.10`. |

## Example Ablation Runs

Annotation-only baseline:

```bash
python final_soccer.py \
  --video "data/SoccerNet/path/to/match/1_224p.mkv" \
  --labels "data/SoccerNet/path/to/match/Labels-v2.json" \
  --half 1 \
  --profile goals \
  --output outputs/baseline_annotation.mp4 \
  --report outputs/baseline_annotation.json
```

Annotation + audio:

```bash
python final_soccer.py \
  --video "data/SoccerNet/path/to/match/1_224p.mkv" \
  --labels "data/SoccerNet/path/to/match/Labels-v2.json" \
  --half 1 \
  --profile goals \
  --use_audio \
  --output outputs/audio_ablation.mp4 \
  --report outputs/audio_ablation.json
```

Annotation + all modalities:

```bash
python final_soccer.py \
  --video "data/SoccerNet/path/to/match/1_224p.mkv" \
  --labels "data/SoccerNet/path/to/match/Labels-v2.json" \
  --half 1 \
  --profile goals \
  --use_clip \
  --use_grounding \
  --use_audio \
  --use_flow \
  --output outputs/full_multimodal.mp4 \
  --report outputs/full_multimodal.json
```


## Output Files

The pipeline produces two main outputs:

```text
summary_v2.mp4
summary_report_v2.json
```

The video file contains the final generated highlight summary.

The JSON report contains:

- selected clip labels
- event timestamps
- clip start and end times
- event score
- scene score
- grounding score
- audio score
- optical-flow score
- replay probability
- final fused score
- evaluation metrics


## Method Overview

The system follows the pipeline below:

1. Load a SoccerNet video half and its corresponding annotation file.
2. Parse event labels and timestamps from `Labels-v2.json`.
3. Generate candidate clips around each event timestamp.
4. Apply a selected user profile to weight event types differently.
5. Optionally extract multimodal features:
   - CLIP scene recognition estimates whether sampled frames resemble soccer concepts such as goal celebrations, attacking play, defensive action, set pieces, crowd reaction, or replay.
   - Grounding DINO detects relevant objects such as the ball, goal, goalkeeper, referee, and cards.
   - Audio scoring uses RMS energy and onset strength as proxies for crowd/commentator excitement.
   - Optical flow estimates motion intensity between sampled frames.
6. Fuse the enabled scores into one final relevance score.
7. Select a diverse set of clips using MMR-style selection.
8. Export the selected clips into a single highlight video.
9. Save a raw results report for analysis and dissertation evaluation.

## Evaluation Metrics

The generated report includes the following technical metrics:

- `precision`: proportion of selected clips that contain at least one important event.
- `recall`: proportion of important events covered by the selected summary.
- `f1`: harmonic mean of precision and recall.
- `summary_duration_seconds`: total duration of the generated summary.
- `event_density_per_minute`: number of covered important events per minute of summary video.
- `baseline_random_recall`: recall achieved by randomly placed clips of similar duration.
- `baseline_uniform_recall`: recall achieved by uniformly spaced clips of similar duration.

## Notes and Limitations

- The system depends on SoccerNet annotations to generate candidate clips, so it is not fully annotation-free.
- CLIP and Grounding DINO are used as scoring components rather than as fully trained soccer-specific models.
- The quality of the output depends on the selected profile, annotation quality, clip windows, fusion weights, and available compute.
- SoccerNet videos are not included in this repository due to dataset access restrictions.

## Author

Tien Nam Nguyen

Dissertation project: Personalised Sports Video Summarisation System Using Multimodal Learning
