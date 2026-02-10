import os
import json
import shutil
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Any

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw):
        print(f"  [{kw.get('desc', '')}] processing...")
        return x


class RallyAwareDatasetSplitter:
    """Split shuttle detection datasets by rally, not by arbitrary frame chunks."""

    @staticmethod
    def detect_rallies(
        annotations: Dict[str, Any],
        sorted_frames: List[str],
        min_rally_length: int = 10,
        max_gap: int = 5,
    ) -> List[List[str]]:
        """Detect rally boundaries from annotations.
        
        A rally is a continuous sequence where the shuttle is visible/occluded
        with gaps ≤ max_gap frames (shuttle briefly leaves frame or is occluded).
        
        Args:
            annotations: Per-frame metadata from annotator
            sorted_frames: Frame names in temporal order
            min_rally_length: Rallies shorter than this are discarded
            max_gap: Max consecutive not_visible frames before ending rally
        
        Returns:
            List of rallies, each rally is a list of frame names
        """
        rallies = []
        current_rally = []
        gap_count = 0
        
        for frame in sorted_frames:
            ann = annotations.get(frame, {})
            vis = ann.get("visibility", "not_visible")
            
            if vis in ("visible", "occluded"):
                # Shuttle is in play
                current_rally.append(frame)
                gap_count = 0
            else:
                # Shuttle not visible
                gap_count += 1
                
                if gap_count <= max_gap and current_rally:
                    # Small gap — shuttle might reappear (high lob, net bounce)
                    current_rally.append(frame)
                else:
                    # Rally ended
                    if len(current_rally) >= min_rally_length:
                        rallies.append(current_rally)
                    current_rally = []
                    gap_count = 0
        
        # Catch last rally
        if len(current_rally) >= min_rally_length:
            rallies.append(current_rally)
        
        return rallies
    
    @staticmethod
    def characterize_rally(rally: List[str], annotations: Dict) -> Dict:
        """Extract rally-level features for stratification.
        
        Returns:
            {
              'length': int,
              'blur_severe_ratio': float,    # % frames with severe blur
              'blur_motion_ratio': float,    # % frames with motion blur
              'occlusion_ratio': float,      # % occluded frames
              'obb_ratio': float,            # % using oriented boxes
              'difficulty': str,             # 'easy' | 'medium' | 'hard'
            }
        """
        blur_counts = defaultdict(int)
        vis_counts  = defaultdict(int)
        type_counts = defaultdict(int)
        
        for frame in rally:
            ann = annotations.get(frame, {})
            blur_counts[ann.get("blur", "clear")] += 1
            vis_counts[ann.get("visibility", "visible")] += 1
            type_counts[ann.get("type", "point")] += 1
        
        n = len(rally)
        severe_ratio = blur_counts["severe_blur"] / n
        motion_ratio = blur_counts["motion_blur"] / n
        occl_ratio   = vis_counts["occluded"] / n
        obb_ratio    = type_counts["oriented_box"] / n
        
        # Heuristic difficulty score
        if severe_ratio > 0.3 or occl_ratio > 0.4:
            difficulty = "hard"
        elif motion_ratio > 0.4 or obb_ratio > 0.3:
            difficulty = "medium"
        else:
            difficulty = "easy"
        
        return {
            "length": n,
            "blur_severe_ratio": severe_ratio,
            "blur_motion_ratio": motion_ratio,
            "occlusion_ratio": occl_ratio,
            "obb_ratio": obb_ratio,
            "difficulty": difficulty,
        }
    
    @staticmethod
    def split_by_rally(
        images_dir: str,
        annotation_output_dir: str,
        output_base_dir: str,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        stratify_by: str = "difficulty",  # 'difficulty' | 'length' | 'none'
        min_rally_length: int = 10,
        max_gap: int = 5,
        seed: int = 42,
    ):
        """Rally-aware split: entire rallies go into one split (conservative).
        
        Best for:
          • Evaluating generalization to unseen game situations
          • Preventing temporal leakage entirely
          • Realistic deployment testing
        
        Drawback:
          • Smaller effective dataset (some rallies too short to use)
        """
        # ── Load annotations ──────────────────────────────────────────────
        ann_path = os.path.join(annotation_output_dir, "annotations.json")
        with open(ann_path) as f:
            annotations = json.load(f)
        
        frames = sorted(annotations.keys())
        print(f"Total annotated frames: {len(frames)}")
        
        # ── Detect rallies ────────────────────────────────────────────────
        rallies = RallyAwareDatasetSplitter.detect_rallies(
            annotations, frames, min_rally_length, max_gap
        )
        print(f"Detected rallies: {len(rallies)}")
        print(f"  Avg length: {np.mean([len(r) for r in rallies]):.1f} frames")
        print(f"  Coverage: {sum(len(r) for r in rallies)} / {len(frames)} "
              f"({sum(len(r) for r in rallies)/len(frames)*100:.1f}%)")
        
        # ── Stratify rallies ──────────────────────────────────────────────
        rng = np.random.default_rng(seed)
        
        if stratify_by == "difficulty":
            strata = defaultdict(list)
            for rally in rallies:
                char = RallyAwareDatasetSplitter.characterize_rally(rally, annotations)
                strata[char["difficulty"]].append(rally)
            
            print("\nDifficulty distribution:")
            for diff in sorted(strata):
                print(f"  {diff:10s} {len(strata[diff]):>4d} rallies")
        
        elif stratify_by == "length":
            strata = defaultdict(list)
            for rally in rallies:
                length_bin = "short" if len(rally) < 30 else "medium" if len(rally) < 80 else "long"
                strata[length_bin].append(rally)
            
            print("\nLength distribution:")
            for lb in ("short", "medium", "long"):
                if lb in strata:
                    print(f"  {lb:10s} {len(strata[lb]):>4d} rallies")
        
        else:  # no stratification
            strata = {"all": rallies}
        
        # ── Split each stratum ────────────────────────────────────────────
        train_rallies, val_rallies, test_rallies = [], [], []
        
        for key, group in strata.items():
            rng.shuffle(group)
            n = len(group)
            
            if n < 3:
                # Too few to split — put in train
                train_rallies.extend(group)
                print(f"  {key}: {n} rallies → all train (too few)")
                continue
            
            tr_n = max(1, int(n * train_ratio))
            va_n = max(1, int(n * val_ratio))
            te_n = n - tr_n - va_n
            
            if te_n < 1:
                te_n = 1
                va_n = max(1, n - tr_n - te_n)
                tr_n = n - va_n - te_n
            
            train_rallies.extend(group[:tr_n])
            val_rallies.extend(group[tr_n:tr_n + va_n])
            test_rallies.extend(group[tr_n + va_n:])
            
            print(f"  {key:10s} {n:>4d} rallies → "
                  f"train {tr_n:>3d}  val {va_n:>3d}  test {te_n:>3d}")
        
        # ── Flatten rallies → frame lists ─────────────────────────────────
        train_frames = [f for r in train_rallies for f in r]
        val_frames   = [f for r in val_rallies   for f in r]
        test_frames  = [f for r in test_rallies  for f in r]
        
        print(f"\nFrame counts:")
        print(f"  Train: {len(train_frames):>5d} frames  ({len(train_rallies)} rallies)")
        print(f"  Val  : {len(val_frames):>5d} frames  ({len(val_rallies)} rallies)")
        print(f"  Test : {len(test_frames):>5d} frames  ({len(test_rallies)} rallies)")
        
        # ── Copy files ────────────────────────────────────────────────────
        RallyAwareDatasetSplitter._copy_splits(
            train_frames, val_frames, test_frames,
            images_dir, annotation_output_dir, output_base_dir,
            annotations
        )
        
        # ── Save metadata ─────────────────────────────────────────────────
        info = {
            "split_method": "rally_aware",
            "stratify_by": stratify_by,
            "min_rally_length": min_rally_length,
            "max_gap": max_gap,
            "seed": seed,
            "rallies_detected": len(rallies),
            "rallies_used": len(train_rallies) + len(val_rallies) + len(test_rallies),
            "counts": {
                "train": {"frames": len(train_frames), "rallies": len(train_rallies)},
                "val":   {"frames": len(val_frames),   "rallies": len(val_rallies)},
                "test":  {"frames": len(test_frames),  "rallies": len(test_rallies)},
            },
        }
        
        with open(os.path.join(output_base_dir, "split_info.json"), "w") as f:
            json.dump(info, f, indent=2)
        
        print(f"\n✓ Rally-aware split complete → {output_base_dir}")
    
    @staticmethod
    def split_by_sequence(
        images_dir: str,
        annotation_output_dir: str,
        output_base_dir: str,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        sequence_length: int = 16,     # frames per training sequence
        sequence_stride: int = 8,      # overlap between sequences
        min_rally_length: int = 10,
        max_gap: int = 5,
        seed: int = 42,
    ):
        """Sequence-based split: rallies are cut into overlapping sequences,
        then sequences are split (more data-efficient).
        
        Best for:
          • Temporal models (TrackNet, LSTM, Transformers)
          • Maximum data utilization
          • Learning motion patterns across frames
        
        How it works:
          1. Detect rallies
          2. Slice each rally into overlapping sequences of `sequence_length`
          3. Split sequences across train/val/test
          4. A rally's sequences may span multiple splits, but NO sequence
             appears in >1 split (partial temporal isolation)
        
        Drawback:
          • Not fully leak-proof (val might see frames 50-65 of a rally
            while train saw frames 0-15 and 30-45)
        """
        # ── Load & detect rallies ─────────────────────────────────────────
        ann_path = os.path.join(annotation_output_dir, "annotations.json")
        with open(ann_path) as f:
            annotations = json.load(f)
        
        frames = sorted(annotations.keys())
        rallies = RallyAwareDatasetSplitter.detect_rallies(
            annotations, frames, min_rally_length, max_gap
        )
        
        print(f"Detected rallies: {len(rallies)}")
        
        # ── Slice rallies into sequences ──────────────────────────────────
        sequences = []
        for rally in rallies:
            if len(rally) < sequence_length:
                # Rally too short for even one sequence — skip or pad?
                # For now: skip. Could also zero-pad.
                continue
            
            for i in range(0, len(rally) - sequence_length + 1, sequence_stride):
                seq = rally[i:i + sequence_length]
                sequences.append(seq)
        
        print(f"Total sequences: {len(sequences)}  (length={sequence_length}, stride={sequence_stride})")
        
        # ── Split sequences ───────────────────────────────────────────────
        rng = np.random.default_rng(seed)
        rng.shuffle(sequences)
        
        n = len(sequences)
        tr_n = int(n * train_ratio)
        va_n = int(n * val_ratio)
        
        train_seqs = sequences[:tr_n]
        val_seqs   = sequences[tr_n:tr_n + va_n]
        test_seqs  = sequences[tr_n + va_n:]
        
        # ── Flatten to frames (a frame may appear in multiple sequences
        #    within the same split, but never across splits) ───────────────
        train_frames = sorted(set(f for seq in train_seqs for f in seq))
        val_frames   = sorted(set(f for seq in val_seqs   for f in seq))
        test_frames  = sorted(set(f for seq in test_seqs  for f in seq))
        
        print(f"\nSplit:")
        print(f"  Train: {len(train_seqs):>4d} seqs → {len(train_frames):>5d} unique frames")
        print(f"  Val  : {len(val_seqs):>4d} seqs → {len(val_frames):>5d} unique frames")
        print(f"  Test : {len(test_seqs):>4d} seqs → {len(test_frames):>5d} unique frames")
        
        # ── Copy files ────────────────────────────────────────────────────
        RallyAwareDatasetSplitter._copy_splits(
            train_frames, val_frames, test_frames,
            images_dir, annotation_output_dir, output_base_dir,
            annotations
        )
        
        # ── Metadata ──────────────────────────────────────────────────────
        info = {
            "split_method": "sequence_based",
            "sequence_length": sequence_length,
            "sequence_stride": sequence_stride,
            "min_rally_length": min_rally_length,
            "max_gap": max_gap,
            "seed": seed,
            "sequences_total": len(sequences),
            "counts": {
                "train": {"sequences": len(train_seqs), "frames": len(train_frames)},
                "val":   {"sequences": len(val_seqs),   "frames": len(val_frames)},
                "test":  {"sequences": len(test_seqs),  "frames": len(test_frames)},
            },
        }
        
        with open(os.path.join(output_base_dir, "split_info.json"), "w") as f:
            json.dump(info, f, indent=2)
        
        print(f"\n✓ Sequence-based split complete → {output_base_dir}")
    
    # ══════════════════════════════════════════════════════════════════════
    # Internal helpers
    # ══════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def _copy_splits(train_frames, val_frames, test_frames,
                     images_dir, annotation_output_dir, output_base_dir,
                     annotations):
        """Copy images, labels, OBB labels, and per-split annotations.json."""
        yolo_dir = os.path.join(annotation_output_dir, "yolo_labels")
        obb_dir  = os.path.join(annotation_output_dir, "yolo_obb_labels")
        
        for split, frames in [("train", train_frames),
                              ("val",   val_frames),
                              ("test",  test_frames)]:
            for sub in ("images", "labels", "obb_labels"):
                os.makedirs(os.path.join(output_base_dir, split, sub), exist_ok=True)
            
            for frame in tqdm(frames, desc=f"Copying {split}"):
                # image
                src = os.path.join(images_dir, frame)
                dst = os.path.join(output_base_dir, split, "images", frame)
                if os.path.exists(src):
                    shutil.copy2(src, dst)
                
                # labels
                stem = os.path.splitext(frame)[0] + ".txt"
                for src_dir, dst_sub in [(yolo_dir, "labels"), (obb_dir, "obb_labels")]:
                    src_lbl = os.path.join(src_dir, stem)
                    dst_lbl = os.path.join(output_base_dir, split, dst_sub, stem)
                    if os.path.exists(src_lbl):
                        shutil.copy2(src_lbl, dst_lbl)
            
            # per-split annotations.json
            subset = {f: annotations[f] for f in frames if f in annotations}
            with open(os.path.join(output_base_dir, split, "annotations.json"), "w") as out:
                json.dump(subset, out, indent=2)


# ══════════════════════════════════════════════════════════════════════════
# Example usage
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Rally-aware dataset splitting")
    parser.add_argument("--method", choices=["rally", "sequence"], required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--stratify-by", default="difficulty",
                        choices=["difficulty", "length", "none"])
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--sequence-stride", type=int, default=8)
    parser.add_argument("--min-rally-length", type=int, default=10)
    parser.add_argument("--max-gap", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    if args.method == "rally":
        RallyAwareDatasetSplitter.split_by_rally(
            images_dir=args.images,
            annotation_output_dir=args.annotations,
            output_base_dir=args.output,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            stratify_by=args.stratify_by,
            min_rally_length=args.min_rally_length,
            max_gap=args.max_gap,
            seed=args.seed,
        )
    else:  # sequence
        RallyAwareDatasetSplitter.split_by_sequence(
            images_dir=args.images,
            annotation_output_dir=args.annotations,
            output_base_dir=args.output,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            sequence_length=args.sequence_length,
            sequence_stride=args.sequence_stride,
            min_rally_length=args.min_rally_length,
            max_gap=args.max_gap,
            seed=args.seed,
        )