#!/usr/bin/env python3
"""
DRAEM Score + Multi-scale Feature Extraction
==============================================
从 DRAEM 提取：
  - image-level anomaly score
  - pixel-level anomaly map
  - 多尺度 encoder features (b3=64x64, b4=32x32, b6=8x8)

输出：
  outputs/{category}/
    train_good/              # 训练集正常图的特征与分数
    test/                    # 测试集（正常+异常）的特征与分数
    scores.csv               # 汇总
"""
import argparse, os, sys, json, glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../../DRAEM')
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork


class ImageFolderDataset(Dataset):
    """Load images from a directory (recursively)."""
    def __init__(self, root_dir, resize_shape=(256, 256), is_train=False):
        self.image_paths = sorted(
            glob.glob(os.path.join(root_dir, '**', '*.png'), recursive=True) +
            glob.glob(os.path.join(root_dir, '**', '*.jpg'), recursive=True)
        )
        self.resize_shape = resize_shape
        self.is_train = is_train

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = image.resize(self.resize_shape, Image.BILINEAR)
        image = torch.tensor(np.array(image).astype(np.float32) / 255.0).permute(2, 0, 1)
        return {
            'image': image,
            'img_path': img_path,
            'img_name': '/'.join(img_path.split('/')[-3:])  # relative path
        }


def extract_features_and_scores(model, model_seg, dataloader, device, feature_layers=('b3', 'b4', 'b6')):
    """
    For each batch, get:
      - anomaly_score: image-level score
      - anomaly_map: pixel-wise anomaly probability
      - features: dict of multi-scale features
    """
    # Enable feature output in DiscriminativeSubNetwork
    model_seg.out_features = True

    results = []
    with torch.no_grad():
        for batch in dataloader:
            gray_batch = batch['image'].to(device)
            img_paths = batch['img_path']
            img_names = batch['img_name']

            # Run reconstruction
            gray_rec = model(gray_batch)

            # Segment: input = concat(reconstruction, original)
            joined_in = torch.cat([gray_rec.detach(), gray_batch], dim=1)
            out_mask, b2, b3, b4, b5, b6 = model_seg(joined_in)
            out_mask_sm = torch.softmax(out_mask, dim=1)

            # Collect requested features
            feat_map = {
                'b3': b3.detach().cpu(),  # (B, 256, 64, 64)
                'b4': b4.detach().cpu(),  # (B, 512, 32, 32)
                'b6': b6.detach().cpu(),  # (B, 512, 8, 8)
            }

            # Per-image processing
            for i in range(gray_batch.shape[0]):
                # Image-level score: max of averaged anomaly map
                amap = out_mask_sm[i, 1:, :, :].unsqueeze(0)  # (1, 1, H, W)
                amap_avg = F.avg_pool2d(amap, 21, stride=1, padding=21//2)
                image_score = amap_avg.max().item()

                # Pixel-level anomaly map
                pixel_amap = amap[0, 0].cpu().numpy()

                # Multi-scale features for this image
                img_feats = {}
                for layer_name in feature_layers:
                    img_feats[layer_name] = feat_map[layer_name][i].numpy()

                results.append({
                    'img_name': img_names[i],
                    'img_path': img_paths[i],
                    'anomaly_score': image_score,
                    'anomaly_map': pixel_amap,
                    'features': img_feats,
                })

    return results


def save_results(results, output_dir):
    """Save extracted features and scores."""
    os.makedirs(output_dir, exist_ok=True)

    # Save per-image scores as CSV
    import csv
    csv_path = os.path.join(output_dir, 'scores.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['img_name', 'anomaly_score'])
        for r in results:
            writer.writerow([r['img_name'], f"{r['anomaly_score']:.6f}"])

    # Save features and anomaly maps as .npy files
    feat_dir = os.path.join(output_dir, 'features')
    amap_dir = os.path.join(output_dir, 'anomaly_maps')
    os.makedirs(feat_dir, exist_ok=True)
    os.makedirs(amap_dir, exist_ok=True)

    for r in results:
        stem = r['img_name'].replace('/', '_').replace('.png', '').replace('.jpg', '')

        # Save multi-scale features
        feat_dict = {}
        for layer_name, feat in r['features'].items():
            feat_dict[layer_name] = feat
        np.save(os.path.join(feat_dir, f'{stem}_feats.npy'), feat_dict)

        # Save anomaly map
        np.save(os.path.join(amap_dir, f'{stem}_amap.npy'), r['anomaly_map'])

    # Save metadata
    meta_path = os.path.join(output_dir, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump({
            'num_images': len(results),
            'image_names': [r['img_name'] for r in results],
            'scores': [r['anomaly_score'] for r in results],
        }, f, indent=2)

    print(f"Saved {len(results)} results to {output_dir}")
    print(f"  scores.csv: {csv_path}")
    print(f"  features/: {feat_dir}")
    print(f"  anomaly_maps/: {amap_dir}")
    return csv_path


def main():
    parser = argparse.ArgumentParser(description='DRAEM feature extraction for Missing Boundary')
    parser.add_argument('--category', type=str, default='bottle')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--base_model_name', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--feature_layers', type=str, default='b3,b4,b6',
                        help='Comma-separated layer names')
    parser.add_argument('--max_train_images', type=int, default=200,
                        help='Max training images to process')
    args = parser.parse_args()

    device = f'cuda:{args.gpu_id}'
    feature_layers = args.feature_layers.split(',')
    os.makedirs(args.output_dir, exist_ok=True)

    # Load DRAEM models
    run_name = f"{args.base_model_name}_{args.category}_"
    rec_path = os.path.join(args.checkpoint_path, run_name + ".pckl")
    seg_path = os.path.join(args.checkpoint_path, run_name + "_seg.pckl")

    print(f"Loading reconstruction model from: {rec_path}")
    model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
    model.load_state_dict(torch.load(rec_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"Loading segmentation model from: {seg_path}")
    model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2, out_features=True)
    model_seg.load_state_dict(torch.load(seg_path, map_location=device))
    model_seg.to(device)
    model_seg.eval()

    # Process train/good images
    train_dir = os.path.join(args.data_path, args.category, 'train', 'good')
    print(f"\n Processing train/good: {train_dir}")
    train_dataset = ImageFolderDataset(train_dir)
    if len(train_dataset) > args.max_train_images:
        # Random subsample
        indices = np.random.choice(len(train_dataset), args.max_train_images, replace=False)
        train_dataset.image_paths = [train_dataset.image_paths[i] for i in indices]
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=2)

    train_results = extract_features_and_scores(
        model, model_seg, train_loader, device, feature_layers
    )
    train_out = os.path.join(args.output_dir, args.category, 'train_good')
    save_results(train_results, train_out)

    # Process test images
    test_dir = os.path.join(args.data_path, args.category, 'test')
    print(f"\n Processing test: {test_dir}")
    test_dataset = ImageFolderDataset(test_dir)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

    test_results = extract_features_and_scores(
        model, model_seg, test_loader, device, feature_layers
    )
    test_out = os.path.join(args.output_dir, args.category, 'test')
    save_results(test_results, test_out)

    # Summary
    train_scores = [r['anomaly_score'] for r in train_results]
    test_scores = [r['anomaly_score'] for r in test_results]
    print(f"\n===== Summary =====")
    print(f"Train (good): {len(train_results)} images, "
          f"score range [{min(train_scores):.4f}, {max(train_scores):.4f}], "
          f"mean={np.mean(train_scores):.4f}")
    print(f"Test: {len(test_results)} images, "
          f"score range [{min(test_scores):.4f}, {max(test_scores):.4f}], "
          f"mean={np.mean(test_scores):.4f}")

    print(f"\nAll features saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
