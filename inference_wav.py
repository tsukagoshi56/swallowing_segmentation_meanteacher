#!/usr/bin/env python3
import os
import math
import argparse
import torch
import torchaudio
import numpy as np
import json
from pathlib import Path
import logging
from exp_config import EPS, IOU_THRESHOLDS, LABEL_SHORT

# --- 依存関係のインポート ---
from models import EventDetector
from metrics import model_output_to_intervals, calculate_event_metrics

# --- ロガー設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_CONFIG_JSON_FILE = SCRIPT_DIR / "config_FT_real.json"

# --- 推論用設定 (ユーザー環境に合わせて変更してください) ---
MODELS_BASE_DIR = SCRIPT_DIR / "results_classification_comparison_re_re"
INFERENCE_OUTPUT_DIR = SCRIPT_DIR / "output" / "test_comparison_re_re"
EXIST_LABEL_DIR = SCRIPT_DIR / "dataset" / "test_data"

# 手動でWAVを指定したい場合
AUDIO_FILES_TO_PROCESS = []

# 実験設定リスト
EXPERIMENTS = [
    {
        "name": "2class_swallowing_vs_others",
        "ssl_model_name": "microsoft/wavlm-base",
        "feature_type": "raw",
        "architecture": "gru",
        "classes": ["swallowing", "others"],
        "base_dir": MODELS_BASE_DIR,
        "checkpoint_candidates": ["best.pth"],
        "prefer_teacher_state": True,
        "teacher_state_key": "teacher_state_dict"
    },
]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEFAULT_CHECKPOINT_CANDIDATES = [
    "best_teacher.pth", "best.pth", "latest_teacher.pth", "latest.pth",
]

CONFIG_METADATA_KEYS = {
    "name", "base_dir", "checkpoint_candidates", "checkpoint_filename",
    "checkpoint_path", "config_path", "prefer_teacher_state",
    "use_teacher_state", "teacher_state_key", "device_override",
}


def load_base_config_from_json(json_path_str: str) -> dict:
    default_fallback_config = {
        "sr": 16000, "ssl_hop_length": 320, "threshold": 0.5,
        "classes": ["unknown_event"], "feature_type": "raw",
        "architecture": "gru", "default_freeze_fe": False,
        "default_freeze_transformer": False,
        "inference_chunk_sec": 10.0, "inference_overlap_sec": 3.0,
    }
    json_path = Path(json_path_str)
    if not json_path.is_file():
        logger.warning(f"基本設定JSONが見つかりません。デフォルトを使用: {json_path}")
        return default_fallback_config.copy()
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get("base_config", data)
    except Exception as e:
        logger.error(f"Config読み込みエラー: {e}")
        return default_fallback_config.copy()

# JSONロード
BASE_CONFIG = load_base_config_from_json(BASE_CONFIG_JSON_FILE)


def load_additional_config(config_path: Path) -> dict:
    if not Path(config_path).is_file(): return {}
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('base_config', data) if isinstance(data, dict) else data
    except Exception:
        return {}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def find_exist_label_pairs(exist_label_dir: str) -> list:
    exist_dir = Path(exist_label_dir)
    if not exist_dir.exists(): return []
    pairs = []
    for wav_file in exist_dir.glob("*.wav"):
        txt_file = wav_file.with_suffix('.txt')
        pairs.append((str(wav_file), str(txt_file) if txt_file.exists() else None))
    logger.info(f"検出されたWAVファイル: {len(pairs)}個")
    return pairs


def parse_ground_truth_file(txt_path: str) -> list:
    """正解ラベル(TXT)を読み込み [(start, end, label), ...] を返す"""
    events = []
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    events.append((float(parts[0]), float(parts[1]), parts[2]))
    except Exception as e:
        logger.error(f"GT解析失敗: {txt_path}, {e}")
    return events


def convert_gt_events_to_intervals(gt_events: list, classes: list) -> dict:
    """
    正解イベントリストを calculate_event_metrics 用の辞書形式に変換
    """
    label_mapping = {
        'f_sw': 'swallowing', 'w_sw': 'swallowing', 's_sw': 'swallowing',
        'sw': 'swallowing', 'ch': 'chewing', 'sp': 'speech',
        'swallowing': 'swallowing', 'chewing': 'chewing',
        'speech': 'speech', 'others': 'others', 'background': 'others'
    }
    
    gt_intervals = {cls: [] for cls in classes}
    
    for start, end, label in gt_events:
        mapped_label = label_mapping.get(label, label)
        if mapped_label in gt_intervals:
            gt_intervals[mapped_label].append((start, end))
        elif mapped_label not in classes and 'others' in gt_intervals:
            gt_intervals['others'].append((start, end))
            
    return gt_intervals


def run_inference_on_single_wav(
    wav_file_path_str: str,
    model: EventDetector,
    config: dict,
    output_dir: Path,
    experiment_name: str,
    gt_txt_path: str = None
) -> dict:
    """
    単一WAVの推論を行い、予測区間辞書を返す。
    """
    wav_file_path = Path(wav_file_path_str)
    if not wav_file_path.is_file(): return {}

    logger.info(f"処理中: {wav_file_path.name}")

    target_sr = config["sr"]
    hop_length = config["ssl_hop_length"]
    hop_time_sec = hop_length / target_sr
    threshold = config["threshold"]
    classes = config["classes"]
    label_short_map = config.get("label_short", LABEL_SHORT)

    # 音声読み込み
    try:
        wav, original_sr = torchaudio.load(str(wav_file_path))
    except Exception as e:
        logger.error(f"Load Error: {e}")
        return {}

    if original_sr != target_sr:
        wav = torchaudio.transforms.Resample(original_sr, target_sr)(wav)
    if wav.shape[0] > 1: wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)
    total_samples = wav.shape[0]

    if total_samples == 0: return {}

    # スライディングウィンドウ
    chunk_sec = config.get("inference_chunk_sec", 10.0)
    overlap_sec = config.get("inference_overlap_sec", 3.0)
    step_sec = chunk_sec - overlap_sec
    
    chunk_samples = int(chunk_sec * target_sr)
    step_samples = int(step_sec * target_sr)
    
    if total_samples <= chunk_samples:
        num_chunks, step_samples = 1, total_samples
    else:
        num_chunks = math.ceil(max(0, total_samples - chunk_samples) / step_samples) + 1

    total_frames = math.ceil(total_samples / hop_length)
    if total_frames == 0: total_frames = 1
    
    global_probs = np.zeros((len(classes), total_frames), dtype=np.float32)

    model.eval()
    device = next(model.parameters()).device

    for i in range(num_chunks):
        start = i * step_samples
        end = min(start + chunk_samples, total_samples)
        segment = wav[start:end]
        if segment.numel() == 0: continue

        with torch.no_grad():
            inp = segment.unsqueeze(0).to(device)
            try:
                logits = model(inp)
                probs = torch.softmax(logits[0], dim=0).cpu().numpy()
            except Exception:
                continue
        
        n_frames = probs.shape[1]
        global_start_frame = start // hop_length
        
        for c_idx in range(len(classes)):
            for f_idx in range(n_frames):
                g_idx = global_start_frame + f_idx
                if g_idx < total_frames:
                    global_probs[c_idx, g_idx] = max(global_probs[c_idx, g_idx], probs[c_idx, f_idx])

    # 区間変換
    merged_probs = torch.from_numpy(global_probs)
    try:
        detected_intervals = model_output_to_intervals(merged_probs, threshold, hop_time_sec, classes)
    except TypeError:
        # local_rank引数が不要なバージョンの場合
        detected_intervals = model_output_to_intervals(merged_probs, threshold, hop_time_sec, classes)

    # テキスト出力
    out_path = output_dir / f"{experiment_name}_{wav_file_path.stem}.txt"
    out_target_classes = [c for c in classes if c not in ['others', 'background']]
    lines = []
    for c in out_target_classes:
        short = label_short_map.get(c, c)
        if c in detected_intervals:
            for s, e in detected_intervals[c]:
                lines.append(f"{s:.6f}\t{e:.6f}\t{short}")
    lines.sort(key=lambda x: float(x.split("\t")[0]))
    
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

    # ★変更点: 結果区間を返すだけにする（評価は外で行う）
    return {
        "detected_intervals": detected_intervals
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", dest="wav_files", nargs="+", metavar="PATH")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--include-exist-label", action="store_true")
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # ディレクトリ・設定準備
    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else INFERENCE_OUTPUT_DIR
    ensure_dir(out_dir)
    
    if args.config:
        base_config = load_base_config_from_json(args.config)
    else:
        base_config = BASE_CONFIG.copy()

    # ファイル収集
    include_exist = (not args.wav_files) or args.include_exist_label
    wav_txt_pairs = find_exist_label_pairs(EXIST_LABEL_DIR) if include_exist else []
    
    manual_wavs = []
    if args.wav_files:
        for w in args.wav_files:
            p = Path(w).resolve()
            if p.is_file(): manual_wavs.append((str(p), None))
    elif AUDIO_FILES_TO_PROCESS:
        for w in AUDIO_FILES_TO_PROCESS:
            p = Path(w).resolve()
            if p.is_file(): manual_wavs.append((str(p), None))

    all_wavs = wav_txt_pairs + manual_wavs
    if not all_wavs:
        logger.warning("処理対象のWAVファイルがありません。")
        return

    # 実験ループ
    for exp_cfg in EXPERIMENTS:
        exp_name = exp_cfg["name"]
        logger.info(f"\n--- 実験 '{exp_name}' ---")
        
        # 設定マージ
        cur_cfg = base_config.copy()
        if exp_cfg.get("config_path"):
            cur_cfg.update(load_additional_config(exp_cfg["config_path"]))
        cur_cfg.update({k: v for k, v in exp_cfg.items() if k not in CONFIG_METADATA_KEYS})
        cur_cfg.setdefault("device", str(DEVICE))
        
        # モデルロード
        base_dir = Path(exp_cfg.get("base_dir", MODELS_BASE_DIR))
        candidates = exp_cfg.get("checkpoint_candidates", DEFAULT_CHECKPOINT_CANDIDATES)
        ckpt_path = None
        for c in candidates:
            p = base_dir / exp_name / c
            if p.is_file():
                ckpt_path = p
                break
        
        if not ckpt_path:
            logger.error(f"Checkpoint not found for {exp_name}")
            continue

        device = torch.device(cur_cfg["device"])
        try:
            model = EventDetector(
                config=cur_cfg,
                ssl_model_name=cur_cfg.get("ssl_model_name"),
                freeze_feature_extractor=cur_cfg.get("freeze_feature_extractor", False),
                freeze_transformer_layers=cur_cfg.get("freeze_transformer_layers", False),
                architecture=cur_cfg.get("architecture"),
                feature_type=cur_cfg.get("feature_type"),
                local_rank=0
            ).to(device)
            
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            # state_dict取得ロジック (簡易化)
            state = ckpt.get(exp_cfg.get("teacher_state_key", "teacher_state_dict")) if exp_cfg.get("prefer_teacher_state") else None
            if state is None: state = ckpt.get("model_state_dict", ckpt)
            state = {k.replace("module.", ""): v for k, v in state.items()}
            model.load_state_dict(state)
            model.eval()
        except Exception as e:
            logger.error(f"モデル初期化失敗: {e}")
            continue

        # ★★★ データセット評価用リストの初期化 ★★★
        dataset_gt_list = []
        dataset_pred_list = []
        
        valid_files_count = 0

        # 推論実行ループ
        for wav, txt in all_wavs:
            res = run_inference_on_single_wav(wav, model, cur_cfg, out_dir, exp_name)
            pred_intervals = res.get("detected_intervals", {})
            
            # 正解ラベルがある場合のみ評価リストに追加
            if txt and Path(txt).exists():
                gt_events = parse_ground_truth_file(txt)
                if gt_events:
                    gt_intervals = convert_gt_events_to_intervals(gt_events, cur_cfg["classes"])
                    
                    # リストに蓄積 (Dataset-based 計算のため)
                    dataset_gt_list.append(gt_intervals)
                    dataset_pred_list.append(pred_intervals)
                    valid_files_count += 1
        
        # ★★★ データセット全体での一括評価 (Notebookと同じ方式) ★★★
        if dataset_gt_list:
            logger.info(f"\n=== {exp_name} : データセット全体評価 (Micro-average, {valid_files_count} files) ===")
            
            target_cls = [c for c in cur_cfg["classes"] if c not in ['others', 'background']]
            iou_thrs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            
            # ここで一括計算
            metrics = calculate_event_metrics(dataset_gt_list, dataset_pred_list, target_cls, iou_thrs)
            
            # 結果表示
            for iou in sorted(metrics.keys()):
                p, r, f1 = metrics[iou]
                logger.info(f"  IoU > {iou:.1f}: Precision={p:.3f}, Recall={r:.3f}, F1={f1:.3f}")
        else:
            logger.warning("評価可能な正解ラベル付きデータがありませんでした。")
            
            
if __name__ == "__main__":
    main()