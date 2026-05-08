"""
Multimodal Soccer Video Summarisation System.
Before running the script, the required SoccerNet videos and annotation files should
be downloaded using download_soccernet.py. The expected data directory is: data/SoccerNet/ for video and labels

To run (full pipeline):
    python final_soccer.py \\
        --video /path/to/1_224p.mkv \\
        --labels /path/to/Labels-v2.json \\
        --half 1 \\
        --profile goals \\
        --use_clip --use_grounding --use_audio --use_flow \\
        --output highlights_v2.mp4

Note: Running the code requires the SoccerNet dataset, which necessitates access permission requirements.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


# Event-profile weights 
EVENT_WEIGHTS: Dict[str, Dict[str, float]] = {
    "goals": {
        "Goal": 1.00, "Shots on target": 0.85, "Shots off target": 0.65,
        "Penalty": 0.90, "Corner": 0.55, "Direct free-kick": 0.55,
        "Indirect free-kick": 0.45, "Foul": 0.30, "Yellow card": 0.35,
        "Red card": 0.50, "Substitution": 0.15, "Offside": 0.25,
        "Clearance": 0.20, "Throw-in": 0.10, "Ball out of play": 0.05,
        "Kick-off": 0.10,
    },
    "defense": {
        "Clearance": 1.00, "Shots on target": 0.85, "Foul": 0.75,
        "Yellow card": 0.60, "Red card": 0.75, "Offside": 0.65,
        "Corner": 0.55, "Goal": 0.70, "Shots off target": 0.45,
        "Penalty": 0.65, "Direct free-kick": 0.50, "Indirect free-kick": 0.45,
        "Throw-in": 0.25, "Ball out of play": 0.20, "Substitution": 0.10,
        "Kick-off": 0.10,
    },
    "discipline": {
        "Foul": 1.00, "Yellow card": 0.95, "Red card": 1.00,
        "Yellow->red card": 1.00, "Penalty": 0.75, "Direct free-kick": 0.55,
        "Indirect free-kick": 0.50, "Offside": 0.35, "Goal": 0.40,
        "Shots on target": 0.30, "Shots off target": 0.20, "Corner": 0.20,
        "Clearance": 0.20, "Throw-in": 0.10, "Ball out of play": 0.10,
        "Substitution": 0.10, "Kick-off": 0.05,
    },
    "balanced": {
        "Goal": 1.00, "Shots on target": 0.80, "Shots off target": 0.55,
        "Penalty": 0.90, "Corner": 0.50, "Direct free-kick": 0.55,
        "Indirect free-kick": 0.45, "Foul": 0.45, "Yellow card": 0.45,
        "Red card": 0.65, "Yellow->red card": 0.70, "Clearance": 0.45,
        "Offside": 0.35, "Substitution": 0.25, "Throw-in": 0.15,
        "Ball out of play": 0.10, "Kick-off": 0.10,
    },
}


# CLIP prompt sets - idea generation for prompts from Sonnet 4.6
SCENE_PROMPT_SETS: Dict[str, List[str]] = {
    "goal_celebration": [
        "a soccer goal celebration with players celebrating",
        "football players celebrating after scoring a goal",
        "soccer players raising their arms in goal celebration",
    ],
    "attacking_play": [
        "a soccer attacking play near the penalty area",
        "football players attacking towards the goal",
        "soccer striker dribbling toward goal",
    ],
    "shot_on_goal": [
        "a soccer player shooting towards the goal",
        "a football player kicking the ball at goal",
        "a soccer shot on target at the goalkeeper",
    ],
    "defensive_action": [
        "a soccer defensive clearance or tackle",
        "a football defender blocking a shot",
        "soccer player making a defensive interception",
    ],
    "goalkeeper_save": [
        "a soccer goalkeeper making a save",
        "a football goalkeeper diving to stop the ball",
        "goalkeeper catching or punching the ball in soccer",
    ],
    "set_piece": [
        "a soccer corner kick, penalty, or free kick situation",
        "players lining up for a free kick in football",
        "soccer penalty kick being taken",
    ],
    "referee_discipline": [
        "a referee showing a yellow card or red card",
        "football referee holding up a card to a player",
        "soccer referee booking a player with a card",
    ],
    "replay": [
        "a slow motion replay in a soccer broadcast",
        "a television replay of a soccer action",
        "broadcast replay graphics overlay during a soccer match",
    ],
    "wide_tactical_view": [
        "a wide broadcast camera view of a soccer match",
        "aerial view of soccer players on the pitch",
        "tactical overhead view of a football game",
    ],
    "crowd_reaction": [
        "a soccer crowd reacting in a stadium",
        "football fans cheering in the stadium stands",
        "stadium crowd celebrating a soccer goal",
    ],
}

SCENE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "goals": {
        "goal_celebration": 1.00, "shot_on_goal": 0.85,
        "attacking_play": 0.75, "goalkeeper_save": 0.60,
        "set_piece": 0.45, "crowd_reaction": 0.50,
        "replay": 0.15, "wide_tactical_view": 0.10,
    },
    "defense": {
        "defensive_action": 1.00, "goalkeeper_save": 0.90,
        "shot_on_goal": 0.65, "wide_tactical_view": 0.35,
        "replay": 0.15,
    },
    "discipline": {
        "referee_discipline": 1.00, "set_piece": 0.55,
        "crowd_reaction": 0.30, "replay": 0.15,
    },
    "balanced": {
        "goal_celebration": 0.85, "shot_on_goal": 0.80,
        "attacking_play": 0.70, "goalkeeper_save": 0.60,
        "defensive_action": 0.55, "referee_discipline": 0.45,
        "set_piece": 0.45, "crowd_reaction": 0.35,
        "replay": 0.15, "wide_tactical_view": 0.15,
    },
}
# SoccerNet provides data for grounding, but Grounding DINO was explored instead 
# to assess multimodal techniques with scoring key clips
GROUNDING_TERMS = [
    "soccer ball", "football goal", "goalkeeper",
    "soccer player", "referee", "yellow card", "red card",
]
GROUNDING_WEIGHTS: Dict[str, Dict[str, float]] = {
    "goals": {
        "soccer ball": 0.50, "football goal": 1.00,
        "goalkeeper": 0.50, "soccer player": 0.25,
    },
    "defense": {
        "soccer ball": 0.45, "goalkeeper": 0.85,
        "soccer player": 0.35, "football goal": 0.45,
    },
    "discipline": {
        "referee": 1.00, "yellow card": 1.00,
        "red card": 1.00, "soccer player": 0.25,
    },
    "balanced": {
        "soccer ball": 0.50, "football goal": 0.60,
        "goalkeeper": 0.45, "soccer player": 0.30,
        "referee": 0.35, "yellow card": 0.55, "red card": 0.65,
    },
}

# Goal / penalty-area spatial zone (normalised image coordinates)
GOAL_ZONE_X = (0.20, 0.80)   # central horizontal band
GOAL_ZONE_Y = (0.10, 0.75)   # upper three-quarters

# Replay gate: clips with replay_prob above this threshold are down-weighted
REPLAY_CONCEPT = "replay"
REPLAY_THRESHOLD = 0.65
REPLAY_DOWN_WEIGHT = 0.60   # multiply final score by this amount


# Candidate dataclass
@dataclass
class CandidateClip:
    label: str
    half: Optional[int]
    event_time: float
    start: float
    end: float
    event_score: float = 0.0
    context_bonus: float = 0.0     
    scene_score: float = 0.0
    grounding_score: float = 0.0
    audio_score: float = 0.0       
    flow_score: float = 0.0        
    replay_prob: float = 0.0       
    final_score: float = 0.0
    scene_details: Optional[Dict[str, float]] = field(default=None)
    grounding_details: Optional[Dict[str, float]] = field(default=None)


# SoccerNet label parsing
def parse_half(game_time: Optional[str]) -> Optional[int]:
    if not game_time:
        return None
    try:
        return int(str(game_time).split("-")[0].strip())
    except Exception:
        return None


def parse_labels_v2(labels_path: Path, half: Optional[int] = None) -> List[dict]:
    with labels_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    parsed = []
    for ann in data.get("annotations", []):
        label = ann.get("label")
        if not label:
            continue
        ann_half = parse_half(ann.get("gameTime"))
        if half is not None and ann_half is not None and ann_half != half:
            continue
        try:
            event_time = float(ann.get("position", 0)) / 1000.0
        except Exception:
            continue
        parsed.append({"label": label, "half": ann_half, "event_time": event_time, "raw": ann})

    return parsed

def game_context_bonus(event_time: float, video_duration: float, max_bonus: float = 0.12) -> float:
    if video_duration <= 0:
        return 0.0
    return float(max_bonus * min(1.0, event_time / video_duration))


def build_candidates(
    annotations: List[dict],
    video_duration: float,
    profile: str,
    pre_context: float = 8.0,
    post_context: float = 10.0,
) -> List[CandidateClip]:
    weights = EVENT_WEIGHTS[profile]
    candidates: List[CandidateClip] = []

    for ann in annotations:
        label = ann["label"]
        event_time = float(ann["event_time"])

        local_pre = pre_context
        local_post = post_context
        if label in {"Goal", "Shots on target", "Penalty"}:
            local_pre += 6
            local_post += 4
        elif label in {"Red card", "Yellow->red card", "Yellow card", "Foul"}:
            local_pre += 3
            local_post += 5

        start = max(0.0, event_time - local_pre)
        end = min(video_duration, event_time + local_post)
        if end <= start:
            continue

        candidates.append(CandidateClip(
            label=label,
            half=ann.get("half"),
            event_time=event_time,
            start=start,
            end=end,
            event_score=float(weights.get(label, 0.05)),
            context_bonus=game_context_bonus(event_time, video_duration),
        ))

    return candidates

class FrameCache:
    """
    Prevents re-decoding frames that appear in multiple overlapping clips.
    """
    def __init__(self, max_size: int = 512):
        self._store: OrderedDict = OrderedDict()
        self._max = max_size

    def get(self, key: tuple) -> Optional[Image.Image]:
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]
        return None

    def put(self, key: tuple, value: Image.Image) -> None:
        self._store[key] = value
        self._store.move_to_end(key)
        while len(self._store) > self._max:
            self._store.popitem(last=False)


_FRAME_CACHE = FrameCache(max_size=512)


def get_video_duration(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps and fps > 0 and frame_count and frame_count > 0:
        return float(frame_count / fps)
    if shutil.which("ffprobe"):
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
        ]
        out = subprocess.check_output(cmd).decode().strip()
        return float(out)
    raise RuntimeError("Cannot determine video duration. Install ffmpeg/ffprobe.")


def extract_frames(
    video_path: Path,
    start: float,
    end: float,
    num_frames: int = 8,      
) -> List[Image.Image]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    path_str = str(video_path)
    times = np.linspace(start, end, num=max(1, num_frames + 2))[1:-1]
    frames: List[Image.Image] = []

    for t in times:
        ms = round(float(t) * 1000.0)
        key = (path_str, ms)
        cached = _FRAME_CACHE.get(key)
        if cached is not None:
            frames.append(cached)
            continue
        cap.set(cv2.CAP_PROP_POS_MSEC, float(max(0, ms)))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        _FRAME_CACHE.put(key, pil_frame)
        frames.append(pil_frame)

    cap.release()
    return frames


# Optical flow motion-intensity scorer, 
def compute_flow_score(frames: List[Image.Image]) -> float:
    '''
    Computes mean dense-optical-flow magnitude across consecutive frame pairs.
    High motion implies likely action moment.
    Compressed to [0, 1] via tanh (reference: ~10 px/frame during broadcast action).
    '''
    if len(frames) < 2:
        return 0.0

    magnitudes: List[float] = []
    prev_gray: Optional[np.ndarray] = None

    for img in frames:
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            mag = float(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean())
            magnitudes.append(mag)
        prev_gray = gray

    if not magnitudes:
        return 0.0
    return float(np.tanh(np.mean(magnitudes) / 10.0))

# Audio excitement score
class AudioExcitementScorer:
    '''
    Extracts the audio track from the video via ffmpeg, then scores each
    candidate clip on two librosa-derived signals:
    '''
    def __init__(self, video_path: Path, sr: int = 16_000):
        try:
            import librosa
            self._lib = librosa
        except ImportError:
            raise ImportError(
            )

        self.sr = sr
        self.audio: Optional[np.ndarray] = None
        self._rms_max = 1.0
        self._flux_max = 1.0
        self._hop = 512
        self._load_audio(video_path)

    def _load_audio(self, video_path: Path) -> None:
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(tmp_fd)
        try:
            cmd = [
                "ffmpeg", "-y", "-i", str(video_path),
                "-ar", str(self.sr), "-ac", "1", "-vn", tmp_path,
            ]
            result = subprocess.run(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.decode()[-2000:])
            self.audio, _ = self._lib.load(tmp_path, sr=self.sr, mono=True)
        except Exception as exc:
            print(f"[Audio] Warning – could not extract audio ({exc}). Audio scores will be 0.")
            self.audio = None
            return
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        hop = self._hop
        rms = self._lib.feature.rms(y=self.audio, hop_length=hop)[0]
        onset_env = self._lib.onset.onset_strength(y=self.audio, sr=self.sr, hop_length=hop)
        self._rms_max = float(np.percentile(rms, 99)) or 1.0
        self._flux_max = float(np.percentile(onset_env, 99)) or 1.0

    def score_clip(self, start: float, end: float) -> float:
        """Returns audio excitement score in [0, 1] for the clip interval."""
        if self.audio is None:
            return 0.0
        s = int(start * self.sr)
        e = int(end * self.sr)
        segment = self.audio[max(0, s):min(len(self.audio), e)]
        if len(segment) < self.sr // 4:
            return 0.0

        hop = self._hop
        rms = self._lib.feature.rms(y=segment, hop_length=hop)[0]
        onset_env = self._lib.onset.onset_strength(y=segment, sr=self.sr, hop_length=hop)

        rms_score = float(np.mean(rms)) / self._rms_max
        flux_score = float(np.mean(onset_env)) / self._flux_max
        return float(np.clip(0.60 * rms_score + 0.40 * flux_score, 0.0, 1.0))

# CLIP scene recognizer  (cosine similarity + sigmoid + ensembling)
class ClipSceneRecognizer:
    def __init__(
        self,
        model_id: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
    ):
        import torch
        import torch.nn.functional as F
        from transformers import CLIPModel, CLIPProcessor

        self._torch = torch
        self._F = F
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.model.eval()

        self.concept_names = list(SCENE_PROMPT_SETS.keys())

        # Build flat prompt list and track which concept each belongs to
        self._all_prompts: List[str] = []
        self._prompt_to_concept: List[int] = []
        for c_idx, name in enumerate(self.concept_names):
            for prompt in SCENE_PROMPT_SETS[name]:
                self._all_prompts.append(prompt)
                self._prompt_to_concept.append(c_idx)

        # Pre-encode text features once – reused for every clip
        with self._torch.no_grad():
            text_in = self.processor(
                text=self._all_prompts, return_tensors="pt", padding=True
            )
            text_in = {k: v.to(self.device) for k, v in text_in.items()}
            text_feats = self.model.get_text_features(**text_in)
            self._text_feats = self._F.normalize(text_feats, dim=-1)  # [N_prompts, D]

    def score_frames(self, frames: List[Image.Image]) -> Dict[str, float]:
        if not frames:
            return {name: 0.0 for name in self.concept_names}

        with self._torch.no_grad():
            img_in = self.processor(images=frames, return_tensors="pt", padding=True)
            img_in = {k: v.to(self.device) for k, v in img_in.items()}
            img_feats = self.model.get_image_features(**img_in)
            img_feats = self._F.normalize(img_feats, dim=-1)   # [N_images, D]

            # Cosine similarities scaled by logit_scale → [N_images, N_prompts]
            logit_scale = self.model.logit_scale.exp()
            sim = (img_feats @ self._text_feats.T) * logit_scale
            # Sigmoid: independent score per (image, prompt) – no zero-sum
            per_prompt = self._torch.sigmoid(sim).cpu().numpy()   # [N_images, N_prompts]

        # Average across frames, then average across phrasings per concept
        avg_per_prompt = per_prompt.mean(axis=0)   # [N_prompts]
        concept_scores: Dict[str, float] = {}
        for c_idx, name in enumerate(self.concept_names):
            indices = [i for i, ci in enumerate(self._prompt_to_concept) if ci == c_idx]
            concept_scores[name] = float(np.mean([avg_per_prompt[i] for i in indices]))

        return concept_scores


# Grounding DINO scorer with spatial bounding-box analysis
class GroundingDinoScorer:
    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        device: Optional[str] = None,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        self.model.eval()
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.text_prompt = ". ".join(GROUNDING_TERMS) + "."

    @staticmethod
    def _spatial_weight(
        box: Tuple[float, float, float, float],
        term: str,
        img_w: int,
        img_h: int,
    ) -> float:
        """
        Returns a spatial multiplier in [0.7, 1.5] based on detection position.

        Ball, goal, and goalkeeper detections near the centre of the frame
        (approximating the goal / penalty area in a broadcast wide-angle shot)
        are up-weighted.  Referee and card detections are up-weighted anywhere
        because they can appear anywhere on the pitch.
        """
        x1, y1, x2, y2 = box
        cx = ((x1 + x2) / 2) / max(img_w, 1)
        cy = ((y1 + y2) / 2) / max(img_h, 1)
        in_goal_zone = (
            GOAL_ZONE_X[0] <= cx <= GOAL_ZONE_X[1]
            and GOAL_ZONE_Y[0] <= cy <= GOAL_ZONE_Y[1]
        )
        if term in {"referee", "yellow card", "red card"}:
            return 1.20
        if term in {"football goal", "soccer ball", "goalkeeper"} and in_goal_zone:
            return 1.50
        if not in_goal_zone:
            return 0.70
        return 1.00

    def score_frames(self, frames: List[Image.Image]) -> Dict[str, float]:
        if not frames:
            return {term: 0.0 for term in GROUNDING_TERMS}

        counts = {term: 0.0 for term in GROUNDING_TERMS}

        for image in frames[:4]:   
            w, h = image.size
            with self._torch.no_grad():
                inputs = self.processor(
                    images=image, text=self.text_prompt, return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                target_sizes = self._torch.tensor([[h, w]], device=self.device)
                results = self.processor.post_process_grounded_object_detection(
                    outputs,
                    input_ids=inputs.get("input_ids"),
                    box_threshold=self.box_threshold,
                    text_threshold=self.text_threshold,
                    target_sizes=target_sizes,
                )[0]

            labels_out = results.get("text_labels", results.get("labels", []))
            scores_out = results.get("scores", [])
            boxes_out = results.get("boxes", [])

            for lbl, score, box in zip(labels_out, scores_out, boxes_out):
                lbl_text = str(lbl).lower()
                conf = float(score.detach().cpu()) if hasattr(score, "detach") else float(score)
                coords: Tuple[float, float, float, float] = (
                    tuple(float(v) for v in box.detach().cpu().tolist())  # type: ignore[assignment]
                    if hasattr(box, "detach") else tuple(float(v) for v in box)
                )
                for term in GROUNDING_TERMS:
                    if term.lower() in lbl_text:
                        sw = self._spatial_weight(coords, term, w, h)
                        counts[term] += conf * sw

        # Saturating exponential compression to [0, 1)
        return {term: float(1.0 - math.exp(-v)) for term, v in counts.items()}


# Fusion utilities

def weighted_average(scores: Dict[str, float], weights: Dict[str, float]) -> float:
    total = sum(weights.values())
    if total <= 0:
        return 0.0
    return float(sum(scores.get(k, 0.0) * w for k, w in weights.items()) / total)


def score_candidates(
    candidates: List[CandidateClip],
    video_path: Path,
    profile: str,
    use_clip: bool = False,
    use_grounding: bool = False,
    use_audio: bool = False,
    use_flow: bool = False,
    clip_frames: int = 8,
    w_event: float = 0.45,
    w_scene: float = 0.20,
    w_grounding: float = 0.10,
    w_audio: float = 0.15,
    w_flow: float = 0.10,
) -> List[CandidateClip]:
    clip_model = ClipSceneRecognizer() if use_clip else None
    grounding_model = GroundingDinoScorer() if use_grounding else None
    audio_scorer = AudioExcitementScorer(video_path) if use_audio else None

    for cand in tqdm(candidates, desc="Scoring candidates"):
        frames: List[Image.Image] = []
        if use_clip or use_grounding or use_flow:
            frames = extract_frames(video_path, cand.start, cand.end, num_frames=clip_frames)

        # ── CLIP scene recognition ──
        if clip_model is not None and frames:
            scene_scores = clip_model.score_frames(frames)
            cand.scene_details = scene_scores
            # Replay gate: record probability before folding into scene score
            cand.replay_prob = float(scene_scores.get(REPLAY_CONCEPT, 0.0))
            cand.scene_score = weighted_average(scene_scores, SCENE_WEIGHTS[profile])

        # ── Grounding DINO ──
        if grounding_model is not None and frames:
            grounding_scores = grounding_model.score_frames(frames)
            cand.grounding_details = grounding_scores
            cand.grounding_score = weighted_average(grounding_scores, GROUNDING_WEIGHTS[profile])

        # ── Audio excitement ──
        if audio_scorer is not None:
            cand.audio_score = audio_scorer.score_clip(cand.start, cand.end)

        # ── Optical flow motion intensity ──
        if use_flow and len(frames) >= 2:
            cand.flow_score = compute_flow_score(frames)

        # ── Replay gate: down-weight clips with high replay probability ──
        replay_factor = REPLAY_DOWN_WEIGHT if cand.replay_prob >= REPLAY_THRESHOLD else 1.0

        # ── Normalised weighted fusion ──
        # The context_bonus is treated as a fraction of the event channel weight.
        active: Dict[str, float] = {
            "event": w_event,
        }
        if use_clip:
            active["scene"] = w_scene
        if use_grounding:
            active["grounding"] = w_grounding
        if use_audio:
            active["audio"] = w_audio
        if use_flow:
            active["flow"] = w_flow

        total_w = sum(active.values())
        raw = (
            active["event"]                    * cand.event_score
            + active.get("scene", 0.0)         * cand.scene_score
            + active.get("grounding", 0.0)     * cand.grounding_score
            + active.get("audio", 0.0)         * cand.audio_score
            + active.get("flow", 0.0)          * cand.flow_score
        ) / total_w

        cand.final_score = float(raw * replay_factor)

    return candidates

# MMR-style diverse clip selection

def mmr_select(
    candidates: List[CandidateClip],
    max_summary_seconds: float = 120.0,
    top_k: int = 12,
    min_gap: float = 5.0,
    lambda_mmr: float = 0.70,
) -> List[CandidateClip]:
    if not candidates:
        return []

    def overlaps(a: CandidateClip, b: CandidateClip) -> bool:
        return not (a.end + min_gap <= b.start or b.end + min_gap <= a.start)

    def label_redundancy(cand: CandidateClip, selected_labels: List[str]) -> float:
        if not selected_labels:
            return 0.0
        return sum(1 for lbl in selected_labels if lbl == cand.label) / len(selected_labels)

    remaining = list(candidates)
    selected: List[CandidateClip] = []
    selected_labels: List[str] = []
    total_duration = 0.0

    while remaining and len(selected) < top_k:
        best: Optional[CandidateClip] = None
        best_mmr = float("-inf")

        for cand in remaining:
            duration = cand.end - cand.start
            if duration <= 0:
                continue
            if total_duration + duration > max_summary_seconds:
                continue
            if any(overlaps(cand, prev) for prev in selected):
                continue

            redundancy = label_redundancy(cand, selected_labels)
            mmr_score = lambda_mmr * cand.final_score - (1.0 - lambda_mmr) * redundancy
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best = cand

        if best is None:
            break

        selected.append(best)
        selected_labels.append(best.label)
        total_duration += best.end - best.start
        remaining.remove(best)

    return sorted(selected, key=lambda c: c.start)

# Evaluation : F1 score, precision, recall, baselines

def _count_covered_events(clips: List[CandidateClip], events: List[dict]) -> int:
    return sum(
        1 for ann in events
        if any(clip.start <= float(ann["event_time"]) <= clip.end for clip in clips)
    )


def _random_clips(video_duration: float, n: int, dur: float) -> List[CandidateClip]:
    import random
    clips = []
    for _ in range(n):
        s = random.uniform(0.0, max(0.0, video_duration - dur))
        clips.append(CandidateClip(
            label="random", half=None, event_time=s + dur / 2,
            start=s, end=min(video_duration, s + dur),
        ))
    return clips


def _uniform_clips(video_duration: float, n: int, dur: float) -> List[CandidateClip]:
    clips = []
    step = video_duration / max(1, n)
    for i in range(n):
        s = i * step
        clips.append(CandidateClip(
            label="uniform", half=None, event_time=s + dur / 2,
            start=s, end=min(video_duration, s + dur),
        ))
    return clips


def evaluate_coverage(
    selected: List[CandidateClip],
    annotations: List[dict],
    profile: str,
    video_duration: float,
    important_threshold: float = 0.50,
) -> Dict[str, float]:
    weights = EVENT_WEIGHTS[profile]
    important = [
        ann for ann in annotations
        if weights.get(ann["label"], 0.0) >= important_threshold
    ]

    # ── Main system metrics ──
    covered = _count_covered_events(selected, important)
    sel_dur = sum(max(0.0, c.end - c.start) for c in selected)
    recall = covered / len(important) if important else 0.0

    # Precision: clips that contain at least one important event
    clips_with_event = sum(
        1 for clip in selected
        if any(
            clip.start <= float(ann["event_time"]) <= clip.end
            for ann in important
        )
    )
    precision = clips_with_event / len(selected) if selected else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    density = covered / (sel_dur / 60.0) if sel_dur > 0 else 0.0

    # ── Baselines ──
    mean_dur = float(np.mean([c.end - c.start for c in selected])) if selected else 15.0
    n = len(selected) or 1

    rand_covered = _count_covered_events(_random_clips(video_duration, n, mean_dur), important)
    unif_covered = _count_covered_events(_uniform_clips(video_duration, n, mean_dur), important)
    rand_recall = rand_covered / len(important) if important else 0.0
    unif_recall = unif_covered / len(important) if important else 0.0

    return {
        "num_important_events": float(len(important)),
        "num_covered": float(covered),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "summary_duration_seconds": float(sel_dur),
        "event_density_per_minute": float(density),
        "baseline_random_recall": float(rand_recall),
        "baseline_uniform_recall": float(unif_recall),
    }

# Export with ffmpeg  

def _run_cmd(cmd: List[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError("Command failed:\n" + " ".join(cmd) + "\n\n" + proc.stderr[-4000:])


def export_summary(video_path: Path, clips: List[CandidateClip], output_path: Path) -> None:
    if not clips:
        raise ValueError("No clips selected; cannot export a summary video.")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required for export. Install it and rerun.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="soccer_sum_v2_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        segment_paths: List[Path] = []

        for i, clip in enumerate(tqdm(clips, desc="Exporting segments")):
            seg = tmpdir_path / f"seg_{i:03d}.mp4"
            _run_cmd([
                "ffmpeg", "-y",
                "-ss", f"{clip.start:.3f}", "-to", f"{clip.end:.3f}",
                "-i", str(video_path),
                "-vf", "scale=1280:-2",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-c:a", "aac", "-b:a", "128k",
                str(seg),
            ])
            segment_paths.append(seg)

        concat_txt = tmpdir_path / "concat.txt"
        with concat_txt.open("w", encoding="utf-8") as f:
            for seg in segment_paths:
                f.write(f"file '{seg.as_posix()}'\n")

        _run_cmd([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_txt), "-c", "copy", str(output_path),
        ])


# Raw data

def save_report(
    clips: List[CandidateClip],
    eval_report: Dict[str, float],
    report_path: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"selected_clips": [asdict(c) for c in clips], "evaluation": eval_report},
            f, indent=2,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multimodal_Video_Summarisation_tnn33",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data ──
    parser.add_argument("--video",   type=Path, required=True,
                        help="Path to SoccerNet half video, e.g. 1_720p.mkv")
    parser.add_argument("--labels",  type=Path, required=True,
                        help="Path to Labels-v2.json")
    parser.add_argument("--half",    type=int, default=None, choices=[1, 2],
                        help="Filter annotations to a single half")
    parser.add_argument("--profile", type=str, default="balanced",
                        choices=list(EVENT_WEIGHTS),
                        help="Event-weighting profile")

    # ── Output ──
    parser.add_argument("--output", type=Path, default=Path("summary_v2.mp4"))
    parser.add_argument("--report", type=Path, default=Path("summary_report_v2.json"))

    # ── Clip window ──
    parser.add_argument("--pre_context",  type=float, default=8.0,
                        help="Seconds before event anchor")
    parser.add_argument("--post_context", type=float, default=10.0,
                        help="Seconds after event anchor")

    # ── Selection ──
    parser.add_argument("--max_summary_seconds", type=float, default=120.0)
    parser.add_argument("--top_k",    type=int,   default=12)
    parser.add_argument("--min_gap",  type=float, default=5.0,
                        help="Minimum gap (s) between selected clips")
    parser.add_argument("--lambda_mmr", type=float, default=0.70,
                        help="MMR trade-off: 1.0=pure relevance, 0.0=pure diversity")

    # ── Modality flags ──
    parser.add_argument("--use_clip",      action="store_true",
                        help="Enable CLIP scene recognition (cosine-sim + sigmoid)")
    parser.add_argument("--use_grounding", action="store_true",
                        help="Enable Grounding DINO with spatial analysis")
    parser.add_argument("--use_audio",     action="store_true",
                        help="Enable librosa audio excitement scoring")
    parser.add_argument("--use_flow",      action="store_true",
                        help="Enable dense optical-flow motion intensity")
    parser.add_argument("--clip_frames",   type=int, default=8,
                        help="Frames sampled per clip for vision models")

    # ── Fusion weights (ablation-ready) ──
    parser.add_argument("--w_event",      type=float, default=0.45)
    parser.add_argument("--w_scene",      type=float, default=0.20)
    parser.add_argument("--w_grounding",  type=float, default=0.10)
    parser.add_argument("--w_audio",      type=float, default=0.15)
    parser.add_argument("--w_flow",       type=float, default=0.10)

    args = parser.parse_args()

    # ── Pipeline ──
    duration    = get_video_duration(args.video)
    annotations = parse_labels_v2(args.labels, half=args.half)
    candidates  = build_candidates(
        annotations,
        video_duration=duration,
        profile=args.profile,
        pre_context=args.pre_context,
        post_context=args.post_context,
    )
    print(f"Loaded {len(annotations)} annotations → {len(candidates)} candidate clips.")

    candidates = score_candidates(
        candidates,
        video_path=args.video,
        profile=args.profile,
        use_clip=args.use_clip,
        use_grounding=args.use_grounding,
        use_audio=args.use_audio,
        use_flow=args.use_flow,
        clip_frames=args.clip_frames,
        w_event=args.w_event,
        w_scene=args.w_scene,
        w_grounding=args.w_grounding,
        w_audio=args.w_audio,
        w_flow=args.w_flow,
    )

    selected = mmr_select(
        candidates,
        max_summary_seconds=args.max_summary_seconds,
        top_k=args.top_k,
        min_gap=args.min_gap,
        lambda_mmr=args.lambda_mmr,
    )

    print(f"\nSelected {len(selected)} clips:")
    header = (
        f"  {'#':>3}  {'Label':<22}  {'Start':>8}  {'End':>8}  "
        f"{'Score':>6}  {'Audio':>6}  {'Flow':>6}  {'Replay':>7}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, clip in enumerate(selected, 1):
        print(
            f"  {i:>3}. {clip.label:<22}  {clip.start:>8.1f}s  {clip.end:>8.1f}s  "
            f"{clip.final_score:>6.3f}  {clip.audio_score:>6.3f}  "
            f"{clip.flow_score:>6.3f}  {clip.replay_prob:>7.3f}"
        )

    eval_report = evaluate_coverage(
        selected, annotations,
        profile=args.profile,
        video_duration=duration,
    )
    print("\nEvaluation:")
    col = max(len(k) for k in eval_report)
    for key, value in eval_report.items():
        print(f"  {key:<{col}} : {value:.3f}")

    save_report(selected, eval_report, args.report)
    print(f"\nReport  → {args.report}")

    export_summary(args.video, selected, args.output)
    print(f"Summary → {args.output}")


if __name__ == "__main__":
    main()