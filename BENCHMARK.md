# NYU 随机权重纯推理测速

从本仓库根目录执行，使用独立环境。发布模型实际上用原生 PyTorch 实现 KNN/GCN，不需要 README 中的 PyTorch Geometric：

```bash
conda env create -f environment.yaml
conda activate GraphCSPN
ln -s ../../data/nyudepthv2_h5 data/nyudepthv2_h5
python -m pytest tests/test_inference.py -q
python scripts/benchmark_nyu.py --output outputs/benchmark_nyu_new_run
```

已有环境/软链接时不要重复创建；输出目录必须尚不存在。`data/` 中原有 loader 和 JSON 不动，只增加相对软链接；搬到其他目录时只调整链接。

## 模型与口径

- 完整 ResNet34、单次 graph propagation、3 层图网络、16 邻点、96 隐藏通道；内部 graph 为 76×102、7752 个节点。保留全量动态 KNN，不缓存/稀疏化/跳过构图。
- 关闭 ResNet ImageNet 加载，保留固定 gather/scatter 权重和原始相机数值。相机常量改为非持久 GPU buffers，随模型提前放 GPU，排除原前向重复 CPU→GPU 传输；checkpoint state_dict 键不变。
- 使用统一 NYU 228×304、500 点、seed=2023、RGB ImageNet 归一化及逐样本确定性采样；随机权重不报告 RMSE。
- RTX 4060 Ti，FP32、batch=1、关闭 TF32/autocast/compile/CUDA graphs，cuDNN benchmark=False、deterministic=False；CPU threads=1。索引 0、326、653 各预热 100 次、CUDA events 逐帧同步计时 1000 次；记录合并平均/P95、FPS 和同步墙钟时间。
- 输入提前置 GPU，计时排除加载/H2D、真值、损失、指标、日志、可视化。显存另测单次前向的峰值 allocated，包含模型和输入。
- 参数包含冻结参数，共享参数去重；权重 MiB 是纯 FP32 state_dict 文件（含 buffers/序列化开销）。MACs 包含卷积（含转置卷积）、图距离矩阵 matmul，不包含 topk、索引收集、softmax 和逐元素操作，不是完整 FLOPs。

## 输出一致性注意

在随机权重下，原模型重复前向也可能因微小 cuDNN 数值差异改变离散 KNN 邻点，因此不一定 allclose。脚本先记录原前向的重复差异，再仅在 wrapper 验证阶段设 `cudnn.deterministic=True`，比较三个真实样本；**正式测速前恢复 False**，不改变统一计时口径。结果不能当作训练后精度或 checkpoint 部署结果。

每次保存 `results.json`、逐次延迟/GPU 状态、配置、随机 state_dict、源码哈希/完整快照、Git 差异和环境清单。短验证可加 `--warmup 2 --iterations 5 --groups 1`，不能作为正式结果。
