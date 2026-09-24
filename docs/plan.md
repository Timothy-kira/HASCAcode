# 3rd WEAR Challenge @HASCA 2026 — 比赛分析 & 构建计划

## Context
用户要打 Kaggle 比赛 `3rd-wear-dataset-challenge-hasca-2026`（截止 2026-10-12 21:59 UTC，剩约 18 天，每天 5 次提交）。
已完成：Kaggle CLI 配置、比赛页面抓取（`docs/competition_pages.md`）、Kaggle CPU 上的 EDA（`kaggle/eda/`，结论见 `docs/eda_findings.md`）。
所有数据处理/训练都在 **Kaggle CPU notebook** 上跑，本地只写代码并 `kaggle kernels push`。技术报告与本比赛无关，不做。

## 1. 比赛要做什么
- 输入：每个测试样本 = 1 秒窗口 = **单个**加速度计 (50,3)（部位已知：左/右 手臂/腿）+ 同一秒中间 15 帧 VideoMAEv2 特征（文件实际形状 `(N,768,15)`，需转置）。
- 输出：19 类（0=null + 18 种运动）；指标 **macro-F1**。
- 测试集：4 个新受试者 (sbj 22–25)，12,234 窗口；训练：22 人、24 段连续录制、~19h，null 占 40%，其余 18 类均衡。
- 当前 LB：第 1 名 0.911，第 2 名 0.887，~0.86 为第 10 名，0.75 左右是“独立窗口”水平。

## 2. 调研结论（可借鉴的论文 / 公开方案）
| 来源 | 要点 | 借鉴 |
|---|---|---|
| WEAR 原论文 (Bock et al., IMWUT 2024, arXiv 2304.05088) | 惯性与视频互补；拼接融合即有效 | 两模态融合的依据 |
| FAME（去年冠军, doi 10.1145/3714394.3756194, code: FranciscoCalatrava/FAME…） | 左右对称传感器共享编码器 + 对称相似度损失；频域/PCA 特征；**逐通道随机符号翻转**增强是泛化主因 | 镜像对齐 + sign-flip 增强 |
| arXiv 2511.23173（单传感器 in-the-wild） | 左右肢数据合并、传感器旋转/轴翻转增强、角度 & SMV + 统计/分形/频谱特征，HGB+XGB 软投票 | 手工特征 + GBDT baseline |
| arXiv 2408.03947（左右互换 & 上下肢配对） | 左右 swap 做数据增强 | 同上 |
| arXiv 2510.21282（PatchTST 分传感器集成 + test-matched 增强） | 每个部位一个模型，jitter/scale/rotation 增强 | 分部位模型/部位 embedding |
| HARMA 报告 (doi 10.1145/3714394.3756192) | 缓解 null 类主导 | null 概率校准/阈值 |
| GitHub acco-cyber/wear-hasca2026-challenge（本届 public 0.887, 第2） | **测试集是完整 session 被切成 1 秒片段后打乱**；用 VideoMAE 首尾相似度 + 惯性边界把窗口串回时间线，再做带时长约束 (80–250 s) 的链式/图解码；null 权重 ×0.5；视频 PCA-160；受试者内 kNN 标签传播。0.75 → 0.817（链解码）→ 0.874 | **最大增益来源：时间线重建 + 序列解码** |

结论：独立窗口分类大约到 0.75–0.80；拉开差距靠 **“把打乱的测试窗口还原成时间序列，再做时序平滑”**（转导式，只利用测试特征、不用人工标注，符合规则 4b）。

## 3. 构建计划（全部为 Kaggle CPU notebook，代码在 `kaggle/<name>/`，沿用 `kaggle/eda/build_nb.py` + `kernel-metadata.json` 的模式）

### Step A — 数据准备 notebook `kaggle/prep/`（输出为 Kaggle 私有 dataset，供后续 notebook 复用）
- 读 24 个训练文件（`label` 按 str 读，NaN→0；sbj_10 的 IMU NaN 线性插值/丢弃）。
- **模拟测试构造**：每段录制切成不重叠 1 s tile，每个 tile 随机取 1 个部位 → (50,3)；视频取对应 30 帧中间 15 帧（索引 8–22，与 EDA 中测试一致性再核对）。同时保留全部 4 部位版本作增强。
- 存 `train_windows.npz`（imu, video15, loc, sbj, session, t_index, label）。

### Step B — 独立窗口模型 `kaggle/baseline/`
- 特征：IMU 统计/频谱/SMV/角度（左臂 x 轴镜像对齐到右臂）+ 视频 15 帧 mean/std/首尾差 → PCA；部位 one-hot。
- LightGBM，**GroupKFold(5) 按 sbj**，OOF macro-F1；null 类权重/阈值调参。
- 出第一版 `submission.csv`（int 标签，列 `id,target_feature`），`kaggle competitions submit`。目标 public ≥0.75。

### Step C — 深度融合模型 `kaggle/fusion/`（CPU 可训）
- IMU 1D-CNN 分支 + 部位 embedding；视频 15×768 → 线性降维 + 小 Transformer/attention pooling；拼接分类。
- 增强：随机旋转、sign-flip、左右 swap、jitter/scale；模态 dropout。
- 与 LightGBM 在 log 概率空间加权融合。

### Step D — 时间线重建 + 序列解码 `kaggle/timeline/`（核心增益）
- 在训练集上用“模拟测试”验证：打乱同一受试者的窗口，用视频特征 (窗口 i 末帧 ↔ 窗口 j 首帧) 余弦相似度 + 互为最近邻 构建后继链，度量链还原准确率。
- 测试集按 sbj 分组重建链/片段；在链上做 HMM / Viterbi 平滑（转移矩阵、活动时长先验从训练集估计），或对链内概率做滑动平均。
- 受试者内视频 kNN 标签传播作补充。
- 验证：留出受试者的 OOF macro-F1（链解码前后对比）。

### Step E — 集成 & 提交管理
- 每天 ≤5 次提交，记录 `docs/submissions.md`（版本、OOF、public LB）。最终选 OOF 与 LB 都稳的一版。

## 4. 验证方式
- 每个 notebook `kaggle kernels push` → Monitor 等待 `kaggle kernels status` → `kaggle kernels output` 拉日志查看 OOF macro-F1。
- 所有 CV 都按受试者分组，并在“模拟测试构造”的数据上评估，确保与 LB 同分布。
- 提交后用 `kaggle competitions submissions` 查看 public 分数，与 OOF 对比检查是否过拟合。

## 5. 注意
- 公开 GitHub 代码仅作思路参考，不直接复制（规则要求共享代码须在 Kaggle 公开；且需 OSI 许可证）。
- Token 已在对话中暴露，赛后建议重置。
- **只用一个账号 `evelynyang02`**：规则明确禁止一人多账号参赛/提交；第二个账号要挂载比赛数据就得加入比赛，而且在其上跑本比赛代码等于跨账号私下共享代码与数据，都有被取消资格的风险。
  并行提速改为：同一账号同时跑多个 CPU notebook（不同折/不同模型拆开跑），特征做成缓存 dataset 复用。
