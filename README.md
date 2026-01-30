SEER: 基于 CLIP 的语义可控 NeRF
===============================

概述
----
本仓库实现了一个基于 NeRF 和 CLIP 的语义可控 3D 渲染框架。通过 NeRF 进行体渲染，并结合 CLIP 文本与图像对齐来实现基于文本的语义控制。该框架支持以下功能：

- 基于 NeRF 的位置编码与视角依赖颜色渲染
- 粗采样与细采样（分层采样）策略
- 可选的 CLIP 文本-图像对齐损失
- 可选的 多视角一致性正则化
- 支持 Blender 风格的 transforms.json 数据集和 LLFF 数据集
- 提供 训练、评估、单张渲染、相机路径渲染 等功能

目录结构
--------
- `src/nerf_model.py`       带位置编码的 NeRF MLP
- `src/seer_model.py`       语义调制封装
- `src/rendering.py`        采样与体渲染（coarse/fine）
- `src/clip_embedding.py`   CLIP 文本/图像嵌入
- `src/data.py`             Rays 数据集载入
- `src/metrics.py`          PSNR/SSIM 评估
- `scripts/data_preprocess.py`  预处理 Blender/LLFF 为 rays 数据
- `scripts/train.py`        训练入口
- `scripts/test.py`         评估入口（PSNR/SSIM）
- `scripts/generate_images.py` 单张渲染
- `scripts/render_path.py`  相机路径渲染
- `scripts/serve.py`        可选 FastAPI 服务

环境依赖
--------
- Python 3.9+
- PyTorch
- numpy、pillow
- open_clip_torch（推荐）或 openai-clip（可选）
- fastapi + uvicorn（仅 `scripts/serve.py` 需要）

安装示例（GPU）：
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy pillow open_clip_torch
```

数据集来源
----------
支持两类常见 NeRF 数据格式：

1) Blender 风格合成数据集
   - 包含 `transforms.json` 或 `transforms_{split}.json`
   - 每帧含 `transform_matrix` 与 `file_path`

2) LLFF 数据集
   - 包含 `poses_bounds.npy`
   - 图像通常在 `images/` 目录下


数据预处理（生成 Rays Payload）
-------------------------------
为了将 Blender 或 LLFF 数据集转换为 rays 数据格式，需要运行数据预处理脚本，生成适合训练的 rays_train.pt 文件。

Blender：
```
python scripts/data_preprocess.py \
  --dataset-dir <blender_scene> \
  --split train \
  --output outputs/rays_train.pt \
  --text "a wooden chair"
```

LLFF：
```
python scripts/data_preprocess.py \
  --dataset-dir <llff_scene> \
  --dataset-type llff \
  --images-dir images \
  --output outputs/rays_train.pt \
  --text "a garden"
```

训练
----
最小训练：
```
python scripts/train.py \
  --data outputs/rays_train.pt \
  --device cuda
```

启用 coarse+fine、CLIP 和一致性：
```
python scripts/train.py \
  --data outputs/rays_train.pt \
  --device cuda \
  --num-samples 64 \
  --num-importance 64 \
  --coarse-loss-weight 0.5 \
  --consistency-weight 0.1 \
  --clip-loss-weight 0.5 \
  --clip-every 50
```

关键训练参数：
- `--num-samples` / `--num-importance`  粗/细采样点数
- `--clip-loss-weight`                 启用 CLIP 语义引导
- `--clip-downsample`                  CLIP 损失降采样
- `--consistency-weight`               多视角一致性权重

评估
----
使用以下命令评估模型并计算 PSNR/SSIM：

```
python scripts/test.py \
  --checkpoint outputs/seer_epoch_5.pt \
  --data outputs/rays_train.pt \
  --device cuda \
  --compute-ssim
```

单张渲染
-------
使用以下命令根据文本描述渲染单张图像：

```
python scripts/generate_images.py \
  --checkpoint outputs/seer_epoch_5.pt \
  --rays outputs/rays_train.pt \
  --prompt "a wooden chair"
```

相机路径渲染
------------
使用以下命令进行相机路径渲染（生成动画或多个视角）：

```
python scripts/render_path.py \
  --dataset-dir <blender_scene> \
  --checkpoint outputs/seer_epoch_5.pt \
  --prompt "a wooden chair" \
  --output-dir renders
```

复现建议
--------
- 固定随机种子：`PYTHONHASHSEED`、`torch.manual_seed`、`torch.cuda.manual_seed_all`
- 固定采样参数：`--num-samples`、`--num-importance`、`--clip-downsample`
- 保持相同预处理与 near/far 设置

备注
----
- CLIP 引导需要 `open_clip_torch` 或 `openai-clip`。
- 若预处理时指定 near/far，会写入 payload 并在训练/推理时优先使用。
