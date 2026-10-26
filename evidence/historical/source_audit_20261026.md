# LHM 备份代码与论文核对报告

核对日期：2026-10-26。范围：用户提供的最终论文 DOCX、其 PDF、34 页 PPT，以及 E:/EmbodiedAssets/LHM_Backup 中与该实验相关的源码、日志、输入和输出。文档中的操作说明仅作为分析材料，未执行其中安装、覆盖或启动命令。原始代码和论文未修改。

## 报告内容到代码的证据表

以下相对代码路径均以 `E:/EmbodiedAssets/LHM_Backup/code/LHM-run-poster-v7` 为根；行号来自本次读取。结论依据调用链和保存的运行记录，而非界面标签。

| 论文或 PPT 内容 | 实现证据 | 核对结果 |
|---|---|---|
| 冻结 LHM，单张参考图生成可动画 avatar | `app_motion_ms.py:1192` 无梯度入口；`:1706` 调用 `lhm.infer_single_view`，再交给动画渲染 | 符合推理原型；未发现该主流程训练 LHM 的步骤 |
| ZJU 输入、单视图源和另一视图 GT | `scripts/prepare_zju_mocap_for_ureca.py` 读取 annots；`ureca_inputs/zju_313_wk2/manifest.json` 保存 source_cam=0、heldout_cam=1、60 帧、5 个关键帧 | 数据准备代码和输入资产存在 |
| 人工或均匀关键帧选择 | `app_motion_ms.py:789` 的 `_select_best_source_frame`，均匀抽样后以清晰度和非白区域比例打分；manifest 中有 0、15、30、44、59 帧 | 选择函数确实实现，不是复杂视角覆盖算法 |
| 选取关键帧参与主重建 | `app_motion_ms.py:1240` 得到 selected_source_frame，但后续只打印及传给状态表 `:1977`；`:1706` 仍使用主参考图生成的 image | 主界面自动选帧没有改变重建输入。用户手动把某关键帧设为参考图仍可运行 |
| 每个关键帧生成一个 canonical prediction | 主入口只调用一次 `infer_single_view`，没有对选中关键帧批量推理和逐帧导出 canonical GS 的循环 | 部分满足：有单帧能力，无完整多关键帧批处理流水线；PPT 第 13 页的保留说明基本准确 |
| Gaussian prefilter | `engine/pose_estimation/pose_utils/postprocess.py:34` 的 `smplx_gs_smooth`，对姿态 6D 表示、形状和位移滤波 | 已实现；baseline 本身也使用 Gaussian，不能将实验解释为全部平滑与完全 raw 的比较 |
| Pose joint-adaptive EMA | 同文件 `:174`、`:220`，在旋转 6D 表示中平滑，再转回 axis-angle | 已实现，但具体关节系数与 PPT 直觉描述不同，详见下文 |
| Translation velocity EMA | 同文件 `:201`，速度 EMA、积分、末端误差线性校正 | 已实现；论文公式省略了末端校正 |
| 运动指标与 ablation | 同文件 `:358`、`:389`；主入口 `:1311` 起生成 baseline、pose-only、trans-only、full 变体 | 有实现及运行日志；原始数值可以对应论文 |
| Level-1.5 img2img diffusion restoration | `app_motion_ms.py:987`、`:1071` 起加载本地 StableDiffusionImg2ImgPipeline，`:1889` 调用恢复流程 | 已实现，且 v7 日志确认真实 diffusion 处理 30/120 帧；不是仅 unsharp 预览 |
| 四路对比视频 | `app_motion_ms.py:870`，及 final 目录中的四路 MP4 | 已保存且完整解码 120 帧。四路为 source、GT、baseline、stabilized，并不包含 diffusion |
| PSNR、SSIM、foreground PSNR、temporal delta、LPIPS | `app_motion_ms.py:578`、`:732`；`:1911` 起接入 UI | 已实现并保存历史数值；LPIPS 默认均匀抽样最多 20 帧，不是全帧平均 |
| 2D foreground-aligned held-out evaluation | 独立脚本 `scripts/v7_camera_aligned_eval_2d.py` 能分别裁剪预测及 GT 前景再缩放；主界面实际走 `compute_heldout_video_metrics` | 独立能力存在，但论文对应主运行未调用它。主流程是整体 resize 加 GT bbox 同坐标裁剪，不能称为完整前景配准 |
| Confidence-weighted fusion | 独立 `scripts/v7_multikeyframe_render_fusion.py` 支持候选视频评分、best 或 top-3 等权均值；主界面未调用 | 只能称为独立 render/score prototype；不是已接入的 confidence-weighted canonical fusion |
| 真正 per-Gaussian 融合、可保存 fused avatar、3D refinement | v7 主界面明确标为未完成；早期 ureca 版本存在 Gaussian 属性加权函数和 identity diffusion hook | 与论文的 future-work 边界相符；早期函数的存在不证明端到端有效融合 |
| Camera B K/R/T 真实新视角渲染 | 主流程没有将 held-out camera 标定接入 renderer；独立 2D 脚本也明确声明非 K/R/T | 未完成，与论文局限说明一致 |

## 整体判断

备份保留了与论文主要实验对应的版本，最相关的是 `code/LHM-run-poster-v7`。`code/LHM-main-ureca` 是更早版本，不能用它单独代表最终实现。`ureca_code` 主要保留 week1 补丁材料。

以“冻结 LHM 周边的运动稳定化、可选单帧 diffusion 恢复及原型评估框架”为定位，核心代码和历史实验有实质证据支持。以图 1、方法 3.1 的字面主线“多个关键帧预测 → confidence-weighted fusion → fused canonical map”为标准，当前主流程不满足。论文后文虽然承认 canonical fusion 未完成，但前面部分句子仍把概念设计写成了实际执行，应统一时态和完成状态。

这次完成的是源码调用链、历史证据和资产完整性核对，不是重新训练或在 GPU 上复现实验成功的证明。

## 原始数值与论文一致

证据：`E:/EmbodiedAssets/LHM_Backup/poster_results/logs/poster_gradio_v7.log:80`、`:85`、`:199`、`:217`。

| 指标 | baseline 日志值 | stabilized 日志值 | 与论文 |
|---|---:|---:|---|
| jitter_pose | 0.1285019219 | 0.0325448886 | 四舍五入一致，下降约 74.7% |
| jitter_trans | 0.0013237192 | 0.0005296041 | 一致，下降约 60.0% |
| drift_trans | 0.0797322094 | 0.0797322094 | 一致，但名称解释需要修正 |
| deviation_pose | 0.1611207873 | 0.4766274393 | 一致，约 2.96 倍 |
| deviation_trans | 0.0014066116 | 0.0082089286 | 一致，约 5.84 倍 |

| Held-out 指标 | baseline | stabilized | diffusion |
|---|---:|---:|---:|
| PSNR full | 0.6286126 | 0.6287913 | 0.7663602 |
| SSIM full | 0.1006769 | 0.1007151 | 0.1026522 |
| PSNR foreground bbox crop | 0.9211741 | 0.9220119 | 1.0642908 |
| temporal delta error | 0.9257648 | 0.8733716 | 1.1012279 |
| LPIPS，最多 20 个均匀样本 | 0.7546956 | 0.7545831 | 0.7517292 |

论文关于 jitter 降低但 motion deviation 增大，以及 diffusion 略改善单帧指标但 temporal delta 恶化的叙述，与历史日志一致。PSNR 绝对值很低，这些结果只能支持该未标定比较下的观测趋势，不能证明可靠的新视角重建质量提升。

## 应修正的科学表述及实现边界

1. **关键帧选择与重建没有接通。** `selected_source_frame` 不进入 `infer_single_view`。因此“5 帧管理/评分”成立，“5 帧均提供重建证据”不成立。3.1、3.2 和 3.12 的 K 次推理目前更适合写作设计或预期复杂度，而非主界面实测执行。

2. **前景对齐能力和论文对应实验不是同一路径。** 主界面使用 GT 的同一个 bbox 裁剪两幅图，没有分别估计预测前景并对齐，也不是按 mask 像素计算纯 foreground PSNR。应写成 GT-foreground-bounding-box crop comparison。独立 2D 对齐脚本的 temporal delta 使用绝对帧差并除以 255，主界面则比较带符号帧差且保留 0–255 单位；两套 temporal delta 不能直接混用。

3. **时间同步没有充分保证。** 本次完整解码确认源视频和 GT 各为 60 帧、30 fps，而最终恢复及四路视频各为 120 帧、30 fps。主评估直接取 `min(len(pred), len(gt))` 并按数组下标比较，没有时间戳或帧 ID 映射。这不自动证明每帧错位，但不能据此声称已经严格同步；需查明 60→120 的来源并记录映射后才能升级结论。

4. **drift_trans 实际为首尾位移距离。** 代码计算 `norm(trans[-1]-trans[0])`，没有减去 GT，也不是累计误差。正常走动同样会增大该值。速度平滑额外做末端校正，会保留首尾位移，因此该值不变不能独立证明消除了漂移。建议改名 endpoint_displacement，或新增相对真实位移的误差指标。

5. **PPT 第 20 页的关节解释和实现不一致。** 当 UI alpha=0.25，实际 root=0.50、body=0.25、手部索引 25:55=0.175。root 更跟随当前，手部更平滑；并非 PPT 所说“root 更稳、手更灵活”的默认安排。也没有独立的头部、脚踝策略。论文宜写成“按关节组设置固定系数的 EMA”，不能解释成依据实时运动自动调整的算法。

6. **baseline 已有 Gaussian。** 新提取 motion 分支中 baseline 为 Gaussian-only，full 为 Gaussian+joint EMA+velocity EMA；已有 motion 分支避免重复 Gaussian，并固定 use_prefilter=False。界面的 Gaussian 开关不能简单理解为对所有入口完全相同的实验控制。应记录分支及原始 motion 来源，避免混称无平滑 baseline。

7. **Diffusion 只覆盖前 30 帧。** 代码将后续帧原样加入恢复视频。整段视频包含已恢复与未恢复帧，单帧独立处理也不保证时间一致性；建议同时报告恢复子集与全序列指标。历史早期 unsharp_preview 运行应与真实 diffusion 成功记录区分；v7 日志确有真实成功证据。

8. **独立多关键帧候选选择读取 held-out GT。** `v7_multikeyframe_render_fusion.py` 根据 GT PSNR 给候选排名。若未来把它用于最终测试，会使测试视图参与选择，不符合“held-out 仅评估”的协议。当前未发现它接入论文对应主运行，不据此指控已发生测试泄漏；将它纳入实验时应改用独立验证集或仅源视图评分。其 mean 模式是 top-3 等权平均，也不是置信度加权。

9. **最新 JSON 已被其他序列覆盖。** `poster_results/zju_313_wk2/gradio_heldout_metrics/latest_heldout_metrics.json` 实际引用 zju_390_kick_wk2，原因是主界面保存目录写死为 zju_313_wk2。论文 ZJU-313 数值应引用保留的 `poster_gradio_v7.log:217`，不能引用该 latest JSON 作为 313 证据。

## 代码恢复与复现状态

建议将 `code/LHM-run-poster-v7` 作为最终原型恢复的候选起点，将其他目录保留为历史版本，不进行覆盖合并。

已验证：

- 论文 DOCX、15 页 PDF 和 PPT 文本可读取；PDF 第 3 页做了视觉查看，未声称对整篇排版审校。
- 主入口、postprocess 和 scripts 下共 16 个 Python 文件 AST 语法解析通过。只验证语法，不验证依赖导入、权重、CUDA 或推理正确性。
- source/GT 两个视频各完整解码 60 帧；四路和 diffusion 视频各完整解码 120 帧。详细元数据保存于同目录 `video_check.json`。
- 本地保留真实 diffusion 恢复视频和四路视频。

复现仍有缺口：历史日志中的 baseline/stabilized 单独视频位于 `/tmp/tmp5o482suc`，最终目录没有这两个独立文件；四路视频保存了对应画面，但不能替代原分辨率独立视频的精确指标复算。日志中的 Linux、模型和临时路径不能直接在 Windows 使用；本次未验证完整模型权重及运行环境，也未执行 GPU 推理。

## 推荐论文措辞

“We implement a frozen single-reference LHM animation prototype with Gaussian-preprocessed motion, group-specific pose EMA, velocity EMA with endpoint correction, optional frame-wise img2img restoration, and rendered-video comparison against another camera stream. Source-keyframe scoring is implemented but is not connected to the main reconstruction input. Batch multi-keyframe reconstruction, confidence-weighted canonical fusion, and calibrated camera rendering remain future work. The reported held-out metrics use resized images and GT bounding-box crops; strict frame correspondence requires further validation.”

这段明确表达当前已恢复代码的能力，同时保留已有实验证据。概念 pipeline 可以继续展示，但应把未接通和 future 模块在图中明确标出。
