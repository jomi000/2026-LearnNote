# MetaQual（金相智鉴）零基础学习笔记

> **适用读者**：计算机系学生，有 Python 基础但较弱，未学过深度学习。
> **覆盖范围**：第 0 讲 ~ 第 5 讲（上）。
> **编写原则**：术语首次出现时标注「中文（英文/缩写）」并用类比解释；不确定处显式标注。
> **未读取内容**：`models/` 下的权重文件、数据集大文件（仅分析代码、配置、文档、脚本）。

---

## 目录

- [第 0 讲 · 导论：项目在解决什么问题](#第-0-讲--导论项目在解决什么问题)
- [第 1 讲 · 项目架构与数据流](#第-1-讲--项目架构与数据流)
- [第 2 讲 · 数据流水线](#第-2-讲--数据流水线)
- [第 3 讲 · 模型原理（从卷积讲起）](#第-3-讲--模型原理从卷积讲起)
- [第 4 讲 · 训练流程](#第-4-讲--训练流程)
- [第 5 讲（上）· 图像处理流水线：预处理 → 推理 → 后处理](#第-5-讲上--图像处理流水线预处理--推理--后处理)
- [附录 A · 术语表](#附录-a--术语表)
- [附录 B · 易错点清单（含本人订正）](#附录-b--易错点清单含本人订正)
- [附录 C · 尚未学习的模块](#附录-c--尚未学习的模块)
- [附录 D · 存疑 / 不确定之处](#附录-d--存疑--不确定之处)

---

## 第 0 讲 · 导论：项目在解决什么问题

### 0.1 材料学背景（三分钟补齐）

**金相学（Metallography）**：把金属样品打磨、抛光、化学腐蚀，再放到显微镜下观察微观组织。

| 概念 | 是什么 |
|---|---|
| **晶粒（Grain）** | 一块金属由无数小晶体拼成，像马赛克地砖，每个小块是一个晶粒 |
| **晶界（Grain Boundary）** | 晶粒之间被腐蚀液腐蚀出的深色缝隙 |
| **假象 / 缺陷（Artifact / Defect）** | 坑洞、夹杂物（杂质颗粒）、水渍、锈迹、划痕。**注意：假象是制样工艺留下的问题，不是材料本身的问题** |

传统流程：**专家用眼睛看照片打分** → 主观、低效、标准不一。

### 0.2 MetaQual 的定位

> **MetaQual = Meta-evaluation of Quality（质量的元评估）** —— 用 AI 去「评估评估这件事本身」。

一句话概括：**把「专家看图打分」改成「模型看图 → 输出量化指标 → 按规则映射成分数」**。

为什么 AI 适合：这个任务恰好是标准的监督学习结构。

| 环节 | 现实对应 | 机器学习对应 |
|---|---|---|
| 输入 | 显微镜照片 | 图像张量 `(H, W, 3)` |
| 人类输出 | 专家在图上圈出晶粒、标出缺陷 | 标注（实例分割掩码 / 多边形） |
| 人类的结论 | 「还算清晰，划痕较少」 | 评分规则 → 分数 |

### 0.3 两条技术路线（来自 README 与 `docs/roadmap.md`）

| | **主线：实例分割（YOLO-Seg）** | **支线：视觉语言模型 VLM** |
|---|---|---|
| 模型 | YOLOv8-Seg / YOLO26-Seg | Florence-2、moondream2、RAM++、Grounding-DINO、SAM2/3 |
| 模式 | 标注 → 训练 → 部署 | 提示词 → 零/少样本 → 端到端输出 |
| 优点 | **精确、可量化、可复现** | 零样本强，可提效标注 |
| 缺点 | 标注成本高 | **会幻觉，工业场景不可控** |
| 角色 | 生产引擎 | 辅助标注工具 + 下一代探索 |

> ⚠️ **代际混用痕迹（已确认）**：`docs/workflow.md` 讲的是 YOLOv8-Seg；但 `models/` 里已有 `yolo26s_pure_iron_steel20_8class_2560_fp32_best.pt`，`run_yolo26_training.sh` 也明确用 `yolo26s-seg.pt`。两者 API 基本一致（都是 Ultralytics 库），训练脚本甚至复用同一个 `train_yolov8.py`。可统一理解为 **Ultralytics YOLO 实例分割家族**。

### 0.4 必须先掌握的三种「看图的 AI」

| 类型 | 类比 | 输出 | 能分清个体吗 |
|---|---|---|---|
| **目标检测（Object Detection）** | 给每个人画个方框 | 一堆矩形 | ❌ |
| **语义分割（Semantic Segmentation）** | 把所有人的像素涂成红色 | 一整片红色 | ❌ 所有人糊成一团 |
| **实例分割（Instance Segmentation）** | 张三涂红、李四涂蓝 | 每人一个独立色块 | ✅ **能** |

> ✅ **本项目必须用实例分割**：评分要算「每个晶粒内部平均亮度 → 再看这些亮度的差异」，这个计算必须知道「哪些像素属于同一个晶粒」。语义分割做不到。

---

## 第 1 讲 · 项目架构与数据流

### 1.1 目录地图（只需先记 4 个）

| 文件夹 | 餐厅类比 | 职责 |
|---|---|---|
| `yolo_training/` | **后厨 / 研发部** | 处理数据、训练、评分算法、可视化（37 个 .py） |
| `backend/` | **前台 + 收银 + 出票** | FastAPI 服务、数据库、PDF 报告、评分规则 |
| `models/` | **菜谱** | 训练好的权重 `.pt`（本次未读取内容） |
| `static/` + `client/` | **门面 / 点餐 App** | React+Vite 前端 + PySide6 桌面端 |

其余（`docs/`、`scripts/`、`export/`、`serverless_models/`、`deploy/`、`tests/`、`cvat_backup/`、`eval_workbench/`、`conductor/`）第一次学习可忽略。

补充：
- `serverless_models/` = 给 CVAT 用的 **AI 辅助标注**服务（SAM3 / YOLOv8-Seg，基于 Nuclio 无服务器架构）
- `deploy/` = Dockerfile + Windows 一键打包脚本
- `docs/` = 38 篇 Markdown，项目最大的知识资产

### 1.2 完整数据流（5 步精简版）

```
① 拍照片（人）              → 金相显微镜
② 圈出晶粒（人）            → CVAT 标注工具，导出 annotations.xml
③ 整理成机器能读的格式（代码）→ yolo_training/ 的数据脚本
④ 训练（代码）              → 得到权重 .pt（「菜谱」）
⑤ 用权重打分（代码）        → 出可视化图 + 出分数
```

**前三步是「教 AI」，后两步是「用 AI」。**

### 1.3 三条「入口 → 核心逻辑」的追代码路径

#### 路径 A：`app.py`（新手最推荐先读）

```python
# app.py:115-123
image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)          # RGB → BGR
vis_bgr, metrics = score_module.process_single_image(...)    # 调核心
vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)          # BGR → RGB
```

- **第 117 行**：Gradio/PIL/Matplotlib 用 **RGB**，OpenCV 用 **BGR**。通道顺序错了对不上号，图像会发蓝/发橘。
- **第 120 行**：`score_module` = `yolo_training.calculate_structure_score`（`app.py:11` 导入）。核心逻辑被刻意解耦成可复用函数，UI 层只做格式转换。

#### 路径 B：`backend/api_server.py`

```
POST /analyze （第 3861 行 analyze_image）
   ├─ 校验 X-Access-Code Header（默认 8888）
   ├─ asyncio.to_thread(...)   ← 把阻塞的 YOLO 推理丢进线程池，避免卡死事件循环
   ├─ _run_local_with_args()（第 3989 行）
   │     └─ process_single_image(...)   ← 同一个函数
   └─ 写库 AnalysisRecord + 落盘 static/results/ + SSE 推送进度
```

#### 路径 C：Jetson 边缘推理（值得学习的技巧）

Jetson 只跑 YOLO 并回传 JSON；本地机器负责打分。为了让两者复用同一套代码，项目用 `_BoxesProxy / _MasksProxy / _ResultProxy` 把 JSON **包装成「长得像 YOLO results 对象」**的东西，再喂给同一个 `process_single_image`。

> 💡 这个技巧叫 **鸭子类型（Duck Typing）**：「如果它走起来像鸭子、叫起来像鸭子，那它就是鸭子」。Python 不需要类型继承，只要有同名属性（`.boxes.cls`、`.masks.data`）即可。

### 1.4 配置如何驱动代码（三层配置）

| 层 | 载体 | 作用 |
|---|---|---|
| **数据配置** | `*.yaml` | 定义**类别表** + **数据集路径** |
| **算法参数** | `argparse.Namespace` 的 `args` | 置信度、核大小、阈值、权重 |
| **服务配置** | 环境变量 / `.env` | `INFERENCE_MODE`、模型路径、访问码 |

关键示例 `models/grain_grading_plus_defects_dataset.yaml`：

```yaml
path: .
train: train/images
val: val/images
test: test/images
nc: 8
names:
  0: 无划痕
  1: 较少划痕
  2: 较多划痕
  3: 很多划痕
  4: 夹杂物
  5: 坑洞
  6: 珠光体
  7: 渗碳体
```

**要点**：
- `names` 是**有序列表**，下标 = **类别 ID**。模型输出 `class_id=3`，代码查表才知道是「很多划痕」。
- **训练时顺序变了，推理全部错位**。
- `nc: 8` 决定网络最后输出层有 8 个通道。
- 巧妙之处：**4 个「晶粒等级」其实是「按划痕严重度分级的晶粒」**，把最难的任务（逐条划痕分割）降级成了更容易的分类问题。

代码的动态适配（不硬编码 ID，改用关键词匹配）：

```python
# yolo_training/calculate_structure_score.py:1463-1471
grain_keywords = {'划痕', '晶粒', 'grain'}
defect_keywords = {'夹杂物', '坑洞', '孔洞', '渗碳体', '石墨', '球墨', '缺陷', 'defect', 'inclusion'}
grain_classes = []
for cid, cname in id_to_name.items():
    cname_str = str(cname)
    if any(dkw in cname_str for dkw in defect_keywords):
        continue
```

---

## 第 2 讲 · 数据流水线

### 2.1 YOLO-Seg 标签格式

一张图一个同名 `.txt`，一个物体一行：

```
<class_id> <x1_norm> <y1_norm> <x2_norm> <y2_norm> ...
```

- 坐标**归一化**（除以宽/高，落在 0~1）
  - 类比：不说「长 300 像素」，说「占图片宽度的 30%」
  - **好处：图片放大缩小标签不用改**
- 检测格式是 `cls cx cy w h`（4 个数）；**分割格式是一长串多边形顶点**

### 2.2 第一步：`convert_cvat_to_yolo.py`（XML → txt）

CVAT 导出的标注有 4 种形态：

| XML 标签 | 含义 | 脚本处理 |
|---|---|---|
| `polygon` | 多边形 | 直接取点 |
| `ellipse` | 椭圆 | 采样 36 个点近似成多边形 |
| `mask` | RLE 编码位图 | 手工解码 → `findContours` → 取**最大**轮廓 |
| `polyline` | 折线（划痕线） | 加粗成带状 → 取轮廓 |

```python
# yolo_training/convert_cvat_to_yolo.py:406-417
if ann_tag.tag == 'polygon':
    if label in SEGMENTATION_CLASSES:
        points_str = ann_tag.get('points').split(';')
        points = [float(coord) for p in points_str for coord in p.split(',')]

        # Smooth Polygon
        points = smooth_polygon(points, args.smooth_factor)

        yolo_segment = convert_points_to_yolo_segment(points, img_width, img_height)
        if yolo_segment:
            yolo_line = f"{class_id} " + " ".join(map(str, yolo_segment))
            yolo_labels.append(yolo_line)
```

- CVAT 的 `points` 格式是 `"x,y;x,y;..."`，先按 `;` 切再按 `,` 切。
- **第 412 行：Douglas-Peucker 算法**（`cv2.approxPolyDP`）
  - 原理：给定闭合折线和容差 ε，保留首尾连线，找离这条线最远的点，若距离 > ε 则保留该点并递归两段……
  - 作用：**去除人工描绘的锯齿手抖噪声**，`--smooth-factor 0.001` 表示 ε = 0.001 × 周长

**`--preset` 是这个脚本的灵魂**（中文标签 → 数字 ID 的映射表）：

| preset | 映射 |
|---|---|
| `grain_grading_plus_defects` | 无/较少/较多/很多划痕晶粒 = 0~3，夹杂物 = 4，坑洞 = 5 |
| `all` | 杂质 = 0，坑洞 = 1，晶粒 = 2（早期 3 类） |
| `scratches_obb` / `scratches_seg` | 划痕任务的两种标注方式 |

### 2.3 第二步：`prepare_yolo_dataset.py`（洗牌 + 切分 + 生成 yaml）

```python
# yolo_training/prepare_yolo_dataset.py:72-80
random.shuffle(paired_files)

total_count = len(paired_files)
train_count = int(total_count * train_ratio)
val_count = int(total_count * val_ratio)

train_set = paired_files[:train_count]
val_set = paired_files[train_count : train_count + val_count]
test_set = paired_files[train_count + val_count :]
```

- **必须先洗牌**：标注通常按图片顺序/批次导出，不洗牌会导致各子集分布不同 → **分布偏移**。默认 **8 : 1 : 1**。
- 三个子集语义完全不同：**训练集**喂梯度；**验证集**每轮算指标用于调参/早停；**测试集**只在最终评估时碰一次。

### 2.4 第三步：`refine_dataset.py`（本项目必做）

核心思想：**不信任人工标注，用几何规则自动修一遍**。五步流水线：

| 步骤 | 函数 | 干什么 |
|---|---|---|
| 1. 去噪 | `is_complex_noise` | 用**圆形度** `4πA/P²` 剔除细长条、乱标线团 |
| 2. 包含过滤 | `filter_containment` | 晶粒：保留**外大**的；缺陷：保留**内小**的 |
| 3. 重叠过滤 | `filter_overlaps` | 按**形状分数**排序，高分的胜出 |
| 4. 平滑 | `smooth_polygon` | 闭运算填裂缝 + 开运算去毛刺 |
| 5. 形状分过滤 | `filter_by_shape_score` | 再筛一遍不合格晶粒 |

**为什么晶粒和缺陷规则相反？（物理决定）**
- 晶粒是**镶嵌结构**，一个晶粒不可能完全包在另一个里面 → 出现「包含」必是重复标注，**留大的**。
- 缺陷通常**位于晶粒内部**，紧贴晶粒边缘画 → 若留外部的大多边形会把整个晶粒吞掉，**留小的**。

形态学操作核心代码：

```python
# yolo_training/refine_dataset.py:54-63
# 1. Morphological Closing (Fill Gaps)
if closing_kernel_size > 0:
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_kernel_size, closing_kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=iterations)

# 2. Morphological Opening (Smooth & Remove Noise)
if kernel_size > 0:
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=iterations)
```

**形态学（Mathematical Morphology）通俗版**：

| 操作 | 类比 | 效果 |
|---|---|---|
| **腐蚀 Erosion** | 拿橡皮沿白色区域边缘擦掉一圈 | 白色**变小**，细线消失 |
| **膨胀 Dilation** | 拿粗记号笔沿边缘描粗一圈 | 白色**变大**，小洞被填 |
| **开运算 Opening**（先腐后膨） | 擦掉毛刺 → 再描回来 | **去毛刺、断细桥**，大形状基本不变 |
| **闭运算 Closing**（先膨后腐） | 填满裂缝 → 再擦回来 | **填内部裂缝和孔洞**，大形状基本不变 |

> 「先闭后开」= **先把晶粒内部的裂纹补上，再把边缘的毛刺磨掉。**

**形状分数**：
```
Score = 圆形度 (4πA/P²) × 实心度 (面积 / 凸包面积)
```
- 圆形度：同样周长下圆面积最大，越接近圆越接近 1；细长条、线团远小于 1
- 实心度：凸包 = 「拿橡皮筋套住这个形状」，凹陷越多比值越小

**图片级动态面积过滤**（很妙的设计）：

```python
# yolo_training/refine_dataset.py:525-527
ratios = [x['ratio'] for x in image_ratios]
median_ratio = np.median(ratios)
threshold = median_ratio * dynamic_area_threshold
```

- 问题：人工标注一张图往往只标**一部分**晶粒；但 YOLO 训练时**没标的地方会被当作背景来学** → 产生**假阴性监督信号**
- 解法：统计每张图「晶粒面积占比」→ 取**中位数** → 阈值 = 中位数 × 0.8 → **删除覆盖率过低的整张图**
- 用**中位数**而非均值：中位数对离群值稳健，少数极端图带不偏阈值

---

## 第 3 讲 · 模型原理（从卷积讲起）

### 3.1 电脑眼里的图片：三维数字阵列

一张 1536×1024 的彩色照片 = 形状 `(1024, 1536, 3)` 的数组：
- 前两维 = 高、宽，每个位置一个像素
- 第三维 = 3 个通道（R、G、B）
- 每个数字 0~255

> 类比：一张超大的 Excel 表，每个格子 3 个数字。**所谓图像处理，本质就是对这些数字做算术。**

### 3.2 卷积（Convolution）—— 手算演示

**CNN（Convolutional Neural Network，卷积神经网络）** 的名字来源。

> 类比：用手电筒照一张大地图。手电筒照亮的 3×3 小方块叫**卷积核 / 滤波器（Kernel / Filter）**，从左上角开始一格一格滑动扫完整图。

在每个位置做：**对应位置相乘，然后全部加起来**。

**示例**（竖直边缘，左暗右亮）：

```
原图窗口      卷积核        逐项相乘
10  10  10    -1  0  +1    -10  0  +10
10  80  80  × -1  0  +1  = -10  0  +80
10  80  80    -1  0  +1    -10  0  +80
                            合计 = 140  ← 很大的正数
```

**对比**（完全平坦，全是 50）：

```
(−50+0+50) + (−50+0+50) + (−50+0+50) = 0  ← 零
```

| 窗口盖住的区域 | 输出 | 含义 |
|---|---|---|
| 左边暗右边亮（**有竖边**） | **140**（大） | 「这里有边！」 |
| 全一样亮（**平坦**） | **0** | 「这里没东西」 |

> ✅ **CNN 的全部直觉**：这个「滑动小窗口 + 对应相乘再求和」的操作，**在有明暗变化的地方输出大数，在平坦的地方输出接近 0**。扫完整图得到的新图就是一张**边缘图**。

**常见手工卷积核**：

| 核 | 名字 | 效果 |
|---|---|---|
| 全 1/9 的 3×3 | 均值模糊 | 每个像素变周围 9 个的平均 → 图变糊（去噪） |
| `[1 0 -1]` 类 | 边缘检测 | 明暗突变处输出大数 → 画出边线 |
| `[0 -1 0; -1 4 -1; 0 -1 0]` | **拉普拉斯算子** | 检测二阶变化 → **锐度/清晰度**度量（本项目第 1321 行用到） |

### 3.3 特征图 + 池化

- 一层卷积的输出叫**特征图（Feature Map）**，含义不是「像素颜色」而是「这个位置有没有某个特征」
- **池化（Pooling）**：2×2 四个数取最大（或平均）→ 合成一个 → **图尺寸减半**

> 类比：**给地图拍缩略图**。不断缩小，单个像素消失，但「哪有山脉、哪有河」这种大尺度结构越来越清楚。

```
输入 1536×1024×3
 ↓ 卷积+池化
 768×512×64      ← 图变小一半，通道变多（64 个不同的「探测器」）
 ↓ ...
 48×32×512       ← 图很小，但每个位置有 512 个数描述它
```

> ✅ **一句话直觉**：越往后，图越小、通道越多、每个格子看到的范围越大、描述越高级。
> （低级：边缘 → 中级：纹理/角点 → 高级：晶界 /「这像个晶粒」）

### 3.4 三段式：Backbone / Neck / Head

| 部分 | 英文 | 类比 | 干什么 |
|---|---|---|---|
| **骨干网络** | Backbone | 读文献、做调研 | 一堆卷积把图变成多层特征图 |
| **颈部** | Neck | 做综述、交叉引用 | 把大尺度粗略特征与小尺度精细特征**融合**（FPN / PAN） |
| **头部** | Head | 下结论、写答案 | 输出框、类别、掩码 |

**为什么需要 Neck？** 本项目的真实痛点：缺陷模型用 **2560** 分辨率训练（抓极小夹杂物），晶粒模型用 **1536**。小目标在大尺度特征图里容易被池化掉，必须把**浅层高分辨率特征**送回来。

> 💡 对应 roadmap 的真实经验：**「对原始图像进行缩放会影响小目标检测能力 → 暂时弃用图片预处理缩放」**。缩小 = 小目标像素更少 = 检测不到。

### 3.5 YOLO 的核心思想：划格子

YOLO = **You Only Look Once（你只需看一次）**。

| | 两阶段（Faster R-CNN、Mask R-CNN） | 单阶段（YOLO） |
|---|---|---|
| 流程 | ① 先找候选区域 ② 再逐个分类精修 | **一步到位**：整图过一次网络，直接输出所有框和类别 |
| 类比 | 先看目录找章节 → 再翻到那页细读 | **通读一遍全文，边读边划重点** |
| 速度 | 慢 | **快**（可实时） |

**每个格子回答 4 个问题**：

| 问题 | 输出 | 类比 |
|---|---|---|
| ① 我格子里**有没有**物体中心？ | 置信度 0~1 | 「报告！我这儿有情况」 |
| ② 中心**具体在哪**？ | 2 个数（x, y 偏移） | 「在我格子右上角偏 0.3、偏下 0.7」 |
| ③ 框**多大**？ | 2 个数（宽, 高） | 「占半张图宽、三分之一高」 |
| ④ 是**哪一类**？ | 每类一个概率 | 「89% 像晶粒，8% 像杂质」 |

**三种粗细的网格**：粗网格（如 20×20）管大物体，细网格（如 80×80）管小物体，同时做。

### 3.6 YOLO-Seg 的掩码：调色盘 + 配方

笨办法：每个物体单独生成一张全图掩码（100 个物体 = 100 张图）→ 太慢。

YOLO-Seg 用 **Prototype + Coefficient** 机制：

```
① 网络生成 N 张「原型掩码」（如 32 张）
   → 类比：调色盘上预先挤好 32 坨基础颜料，每坨都是一张模糊的整图图案

② 每个检测框额外输出 N 个「系数」
   → 类比：一张配方「取 3 份 1 号 + 0.5 份 7 号 − 2 份 20 号」

③ 最终掩码 = 系数 × 原型掩码 加权求和，裁剪到框内、阈值化
   → 按配方调色，得到该物体专属形状
```

**好处**：不管图上有 5 个还是 500 个物体，只需生成 **32 张共享原型** + 每物体 **32 个数字**。

代码痕迹（`calculate_structure_score.py`）：
- `masks.data`（第 1460 行）：原型组合出的**掩码张量**，分辨率通常不等于原图，需 `cv2.resize` 回 `(w, h)`
- `masks.xy`（第 1413 行）：转成**轮廓多边形点集**，做几何运算更方便
- `retina_masks=True`：让原型掩码按**原图分辨率**生成，轮廓更精细（代价：更慢、更吃显存）

### 3.7 为什么选 YOLO-Seg（项目真实踩过的坑）

| 候选 | 结论 | 原因 |
|---|---|---|
| **Cellpose（cyto3）** | ❌ 终止 | 为**生物细胞**训练。细胞是散落在背景上的独立个体，晶粒是**紧密镶嵌铺满全图** → **领域差异（Domain Gap）** |
| **U-Net** | ⚠️ 早期设想 | `docs/feasibility.md` 提过做晶界分割。但 U-Net 是语义分割，**分不清个体** |
| **DINOv3** | ❌ 终止 | 权重需申请权限（HuggingFace Gated），下载不到 |
| **YOLOv8-Seg** | ✅ **主线** | 精确、可量化、快、Ultralytics 生态成熟（TensorRT 导出、CLI、W&B 现成） |

> 💡 **工程直觉**：选模型不是选「最先进」，而是选「能落地 + 生态全 + 训得动」。

### 3.8 NMS 与两个阈值

> 类比：**一群人投票选班长**。分最高的当选；跟它高度重合（IoU > 阈值）的其他人全部弃权；再在剩下的人里选下一个……

```python
# calculate_structure_score.py:1345
grain_results = model.predict(source=original_image, conf=grain_conf_threshold, iou=0.45,
                              verbose=False, retina_masks=retina_masks, agnostic_nms=True,
                              imgsz=args.imgsz, device=inference_device)
```

| 参数 | 含义 | 本项目值 |
|---|---|---|
| `conf` | 置信度阈值，低于它的框直接丢 | 晶粒 0.3 / 缺陷 0.25 / 划痕 **0.1** |
| `iou` | 重合度阈值，两框重合超此比例只留高分 | 0.45 |

> 划痕 conf 只有 **0.1**：划痕细而难检，**阈值调低是「宁可多检也不漏检」**（牺牲精确率换召回率）。

---

## 第 4 讲 · 训练流程

### 4.1 猜价格游戏（梯度下降的全部直觉）

真实价 3000 元，我来猜：

| 轮次 | 我猜 | 差多少 | 怎么调 |
|---|---|---|---|
| 1 | 5000 | 高了 2000 | 往下调 |
| 2 | 3500 | 高了 500 | 往下调（幅度小一点） |
| 3 | 3100 | 高了 100 | 再小一点 |
| 4 | 3005 | 高了 5 | 几乎对了 |

| 游戏里 | 机器学习里 | 本项目对应 |
|---|---|---|
| 我猜的价格 | 模型的**预测** | 输出的框 + 掩码 |
| 真实价 3000 | **标注（Ground Truth）** | CVAT 里人工圈的多边形 |
| 差多少 | **损失（Loss）** | 框偏多少 + 类别错没错 + 掩码像不像 |
| 往下调 | **更新权重** | 修改卷积核里的数字 |
| 调整幅度 | **学习率（Learning Rate, lr）** | `lr0 = 0.01` |
| 猜了很多轮 | **Epoch（轮次）** | `epochs = 100` |

> ✅ **训练 = 猜 → 看错多少 → 往对的方向微调 → 重复。**

### 4.2 损失函数（Loss Function）

YOLO 的预测错有三种错法，损失是**三项相加**：

| 错在哪 | 通俗说法 | 典型算法 |
|---|---|---|
| **框的位置/大小错** | 框偏了 20 像素 / 框大一圈 | CIoU / DIoU Loss |
| **类别判错** | 把「坑洞」说成「夹杂物」 | 交叉熵 / BCE |
| **掩码形状不对**（Seg 专有） | 轮廓歪了、多块、少块 | BCE（逐像素比对） |

```
总损失 = 框的损失 + 类别的损失 + 掩码的损失
```

> 📌 **本项目重要事实**：项目**没有自己写损失函数**。`train_yolov8.py` 就是一句 `model.train(...)`，损失/优化器/学习率策略全部由 **Ultralytics 库**内置提供。

### 4.3 Batch 与 Epoch

- **Batch**：一次喂几张图。本项目 `--batch 4`（1536~2560 高分辨率，显存吃紧）
- **Epoch**：训练集**所有图过一遍**。本项目 `--epochs 100`

> 类比：**考试对答案**。一次做 1000 道再对答案 → 忘了第 1 题怎么想的（且显存放不下）；**每做 4 道就对一次答案立刻修正** → 学得快、省内存。

**Batch 越小，更新越频繁但越「偏」；Batch 越大，更新越稳但更吃显存。**

### 4.4 过拟合（Overfitting）

> 类比：学生把过去 5 年真题答案全背下来，月考满分；高考出新题 → 当场崩盘。

**怎么发现**：
```
训练集损失 ↓ 一直降，很漂亮
验证集损失 ↓ 降到某点后开始 ↑ 反弹   ← 警报！过拟合
```

**本项目怎么防（真实存在的手段）**：

| 手段 | 在哪 | 原理 |
|---|---|---|
| **数据增强** | `train_yolov8.py` | 人为制造「新题」 |
| **预训练权重（迁移学习）** | `--model yolov8s-seg.pt` | 从「已认识边缘纹理」的权重起步，所需样本量大减 |
| **验证集监控 + 保存 best.pt** | Ultralytics 自动 | 保存**验证集表现最好的那一轮**，而非最后一轮 |
| **多尺度训练** | `--multi-scale` | 每张图随机缩放 0.5~1.5 倍 |

### 4.5 数据增强（Data Augmentation）

> 核心思想：**一张图变出很多张「看起来不一样但答案一样」的图**。
> 类比：给同一张试卷拍照 —— 旋转 90°、调亮、镜像翻转。题没变，但对学生是「新卷子」。

| 参数 | 默认值 | 干了什么 |
|---|---|---|
| `--mosaic` | 1.0 | **马赛克增强**：4 张图拼成 1 张 → 单图样本更多、**小目标变多** |
| `--copy-paste` | 0.0 | **复制粘贴**：抠出物体贴到另一张图 → 对小样本类别特别有效 |
| `--multi-scale` | 可选 | 随机缩放 0.5~1.5 倍 |
| `--amp` | True | **混合精度**：部分计算用 FP16 → **省显存、提速**（注意：省的是显存，不是防过拟合） |

> 💡 roadmap 真实数据：「假象约 2040 个，晶粒约 37500 个，目标补到 12500 个」→ **`--copy-paste` 正是为「类别极度不平衡」准备的武器**。

### 4.6 评测指标

#### ① IoU（交并比）
```
IoU = 两框重叠面积 ÷ 两框总覆盖面积
```
- 1.0 = 完美重合；0.5 = 重叠一半；0 = 完全不沾边
- **两个用途别混淆**：评测时「预测框 vs 标注框」；NMS 时「预测框 A vs 预测框 B」

#### ② Precision（精确率）与 Recall（召回率）
> 类比：**草地上找蘑菇**
- 摘了 10 个，8 个真蘑菇 → **精确率 = 8/10 = 80%**（摘的东西里多少是对的）
- 草地其实有 20 个真蘑菇，只找到 8 个 → **召回率 = 8/20 = 40%**（真的里面找到了多少）

**两者互相拉扯**（多摘能提高召回，但石头也多了，精确率就掉）。

#### ③ mAP（平均精度）
把所有类别的「精确率-召回率」表现平均起来，检测领域的**头号指标**。
- **mAP@0.5**：IoU > 0.5 就算对（宽松）
- **mAP@0.5:0.95**：0.5 到 0.95 取 10 个阈值分别算再平均（严格）

### 4.7 本项目真实训练命令

```bash
python3 yolo_training/train_yolov8.py \
  --data ../yolo_dataset_grain_grading_plus_defects_refined/xxx.yaml \
  --epochs 100 --batch 4 --imgsz 1536 \
  --model yolov8s-seg.pt \
  --wandb-project "MetalQual" --wandb-name "Unified-Grading-Defects-1536-Run39"
```

| 参数 | 对应概念 |
|---|---|
| `--data` | 第 2 讲的 dataset.yaml |
| `--epochs 100` | 猜多少轮 |
| `--batch 4` | 每 4 道对一次答案 |
| `--imgsz 1536` | 输入分辨率（小目标靠高分辨率） |
| `--model yolov8s-seg.pt` | 迁移学习起点 |
| `--wandb-*` | 实验记录（对比 39 次实验的 loss/mAP 曲线） |

> 🔥 **工程细节**：`run_yolo26_training.sh:17` 写了 `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，注释写明「修复高分辨率下显存碎片导致的 CUDA Out of Memory」。

---

## 第 5 讲（上） · 图像处理流水线：预处理 → 推理 → 后处理

**全程拆解「心脏函数」** `_do_process_single_image`（`calculate_structure_score.py` 第 **1294** 行起，被切成 Stage 1~12）。

### 5.1 全局地图

| Stage | 干什么 | 类别 |
|---|---|---|
| 1 | 灰度化 + 拉普拉斯 | **预处理** |
| 2 / 3 | 晶粒模型、缺陷模型推理 | **推理** |
| 4 | 假象（坑洞/水渍/锈迹）计数 | 解析 |
| 5 | 晶粒掩码后处理（六小步） | **后处理**（本讲重点） |
| 6 | 均匀度统计 | 计算 |
| 8~9 | 划痕骨架化 + 曲线拟合 | 后处理 |
| 11 | 清晰度梯度计算 | 计算 |
| 12 | 四宫格可视化拼接 | **可视化** |

### 5.2 Stage 1：预处理（只有两行）

```python
# yolo_training/calculate_structure_score.py:1320-1321
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
laplacian_image = cv2.Laplacian(gray_image, cv2.CV_64F)
```

- **灰度化**：3 通道 → 1 通道。晶界是「明暗」信息，颜色没用；1 通道计算量只有 1/3。
  - 本质是一个固定权重卷积：`0.114×B + 0.587×G + 0.299×R`（人眼对绿最敏感，权重最大）
- **拉普拉斯**：第 3 讲那个核的加强版。**边缘处数值大，平坦处接近 0**
  - `CV_64F` = 64 位浮点，因为结果有负数（0~255 整数存不下）

> ⚠️ **注意：这里没有「去噪」预处理。** roadmap 明确记着「对原始图像缩放会影响小目标检测 → 暂弃用」。项目选择**把降噪交给模型本身和后处理**，不对原图动手。

### 5.3 Stage 2/3：推理

| 模型 | imgsz | conf |
|---|---|---|
| 晶粒模型 | `args.imgsz`（1536） | 0.3 |
| 缺陷模型 | **2560** | 0.25 |

缺陷模型用更高分辨率 = 抓极小夹杂物（对应 3.4 节的 Neck 多尺度）。

### 5.4 Stage 5 核心：一张 `label_map`（本项目最妙的设计）

问题：模型输出的是**一堆各自独立的掩码**，但金相组织是**镶嵌结构**，必须严丝合缝、不能重叠。

办法：造一张**和原图同大的整数图**：

```
label_map[y][x] = 0   → 这个像素不属于任何晶粒（空白）
label_map[y][x] = 7   → 这个像素属于第 7 号晶粒
```

> 类比：**给地图上色，每个国家涂一个编号。** 有了这张图，「谁挨着谁」「哪里是空白」一目了然。

#### 小步 5.1：抢地盘式绘制

```python
# calculate_structure_score.py:1539-1547
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(sub_mask_resized, connectivity=8)
if num_labels > 2:
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    sub_mask_resized = (labels == largest_label).astype(np.uint8)

sub_label_map = label_map[Y1:Y2, X1:X2]
pixels_to_paint = (sub_mask_resized > 0) & (sub_label_map == 0)
sub_label_map[pixels_to_paint] = internal_id
id_to_class[internal_id] = class_id
```

- **1539 行 连通域标记**：一个掩码可能断开成几块，`connectedComponentsWithStats` 把每块分开编号并统计面积
- **1541 行**：**只保留面积最大的那块**，细碎岛直接扔
- **1546 行（关键）**：`pixels_to_paint` 要求 `sub_label_map == 0`，即**只画到空白的地方**
- **1547 行**：盖上自己的编号

> 💡 **为什么按置信度从高到低画？**（第 1477 行 `np.argsort(-conf_vals)`）
> 类比：**抢地盘，把握大的先占。** 置信高的晶粒先圈地，低的只能在剩下空地落脚 → **自动避免重叠**。

#### 小步 5.2：实心度（Solidity）—— 拆开「葫芦形团块」

```python
# calculate_structure_score.py:1579-1598
hull = cv2.convexHull(cnt)
hull_area = cv2.contourArea(hull)

if hull_area > 0:
    solidity = float(area) / hull_area
    if solidity < solidity_threshold:      # 阈值 0.75
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        num_labels_open, labels_open, stats_open, _ = cv2.connectedComponentsWithStats(opened_mask, connectivity=8)
        if num_labels_open > 2:
            largest_label_open = 1 + np.argmax(stats_open[1:, cv2.CC_STAT_AREA])
            opened_mask = (labels_open == largest_label_open).astype(np.uint8)

        sub_label[sub_label == uid] = 0

        core_area = np.count_nonzero(opened_mask)
        if core_area > min_grain_area:
            sub_label[opened_mask > 0] = uid
            locked_ids.add(uid)
```

- **凸包（Convex Hull）**：类比「拿橡皮筋套住这个形状」
- **实心度 = 实际面积 ÷ 凸包面积**：圆 → 1.0；葫芦（中间凹）→ 明显小于 1
- **阈值 0.75**：低于它 = 形状不正常
- **开运算**：「先腐蚀后膨胀」→ **专门断掉葫芦中间那根细「桥」**
- 断完可能又碎 → **只留最大主体**

> 🔑 **形态学操作在真实项目中的教科书级应用**。

#### 小步 5.3：剔除小晶粒

`min_grain_area = int(300 * (scale_factor ** 2))`，其中 `scale_factor = max(1.0, w/1000.0)`。
**随分辨率动态缩放** —— 1080p 下约 300 像素，2560 宽下阈值自动变大。

#### 小步 5.4：Voronoi 填充（把缝隙全部补上）

```python
# calculate_structure_score.py:1626-1633
if np.any(good_grain_mask):
    edt_input = np.logical_not(good_grain_mask)
    distances, indices = distance_transform_edt(edt_input, return_indices=True)
    nearest_y = indices[0]
    nearest_x = indices[1]
    nearest_labels = label_map[nearest_y, nearest_x]
    empty_pixels = (label_map == 0)
    label_map[empty_pixels] = nearest_labels[empty_pixels]
```

**距离变换（Distance Transform）通俗版**：
> 对每个空白像素问：**「离我最近的晶粒像素是哪个？」** 然后把这块空白**划归给它**。

- `return_indices=True` → 返回最近前景的**坐标**（不只是距离）
- 第 1631 行：用这些坐标去 `label_map` 查「最近的晶粒是几号」
- 第 1633 行：把所有空白像素填上最近晶粒的编号

> 类比：**三国打完仗分地盘，无主之地归最近的国家。** 数学上叫 **Voronoi 图**。

#### 小步 5.5：最终形态学平滑 + 二次填充

用 5×5 椭圆核开运算逐个 ROI 抛光，再跑一次距离变换二次填充。

#### 小步 5.6：轮廓提取（整数图 → 多边形）

```python
# calculate_structure_score.py:1684-1694
id_mask = (label_map == uid).astype(np.uint8)
contours, _ = cv2.findContours(id_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not contours: continue
cnt = max(contours, key=cv2.contourArea)
score = id_to_score.get(uid, 0.0)
class_id = id_to_class.get(uid, 0)
cleaned_detections.append({
    'class_id': class_id,
    'polygon': Polygon(cnt.reshape(-1, 2)),
    'score': score
})
```

- **1684 行 = 阈值化**：`label_map == uid` 是布尔判断，转 `uint8` 即 0/1 二值图。这就是「阈值化」，只不过「阈值」是「等于某编号」
- **1685 行 `findContours` = 轮廓提取**：OpenCV 的 Suzuki 边界跟踪算法，沿白色区域边界走一圈输出点串
  - `RETR_EXTERNAL` = 只取**最外层**（忽略内部洞）
  - `CHAIN_APPROX_SIMPLE` = **压缩**：一条直线上只存首尾两点，省内存
- **1692 行**：转成 `shapely.geometry.Polygon`，供后续「几何收缩」「重叠判断」使用

#### 小步 5.7：交叠过滤

`filter_overlapping_masks(..., overlap_threshold=0.8)`：按面积降序，若某小多边形有 >80% 面积落在已接受的大多边形内 → 视为碎片丢弃。

### 5.5 图像处理流水线总表

| 步骤 | 技术 | OpenCV 实现 | 作用 |
|---|---|---|---|
| **灰度化** | 加权求和 | `cvtColor` | 3 通道 → 1 通道 |
| **锐度提取** | 拉普拉斯卷积 | `Laplacian` | 边缘处数值大 |
| **连通域** | 泛洪 / 并查集 | `connectedComponentsWithStats` | 散块分开编号 |
| **去碎片** | 取最大 | `argmax(面积)` | 只留主体 |
| **去细桥** | 开运算 | `MORPH_OPEN` | 断细连接 |
| **填缝** | 距离变换 | `distance_transform_edt` | 无主之地归最近的 |
| **轮廓** | 边界跟踪 | `findContours` | 像素 → 点串 |
| **几何运算** | 计算几何 | `shapely` | 收缩、求交、判包含 |

---

## 附录 A · 术语表

| 中文 | 英文 / 缩写 | 通俗解释 |
|---|---|---|
| 金相学 | Metallography | 打磨腐蚀金属、显微镜看微观组织 |
| 晶粒 / 晶界 | Grain / Grain Boundary | 马赛克小块 / 小块之间的缝 |
| 假象 | Artifact | 制样工艺留下的坑洞、水渍、锈迹（非材料问题） |
| 目标检测 | Object Detection | 找位置、画矩形框 |
| 语义分割 | Semantic Segmentation | 逐像素分类，**不区分个体** |
| **实例分割** | Instance Segmentation | 逐像素分类**且区分个体**（本项目核心） |
| 掩码 | Mask | 与原图同尺寸的二值图，白 = 该物体覆盖的像素 |
| 旋转框检测 | OBB | 带角度的矩形框，适合细长划痕 |
| 卷积核 / 滤波器 | Kernel / Filter | 滑动的小窗口，里面是一组可学习的权重 |
| 特征图 | Feature Map | 一层卷积的输出 |
| 池化 | Pooling | 2×2 取最大 → 图尺寸减半 |
| 骨干 / 颈 / 头 | Backbone / Neck / Head | 提取特征 / 融合多尺度 / 下结论 |
| 损失 | Loss | 「错了多离谱」的量化值 |
| 学习率 | Learning Rate（**lr**） | 每次往对的方向调整的幅度 |
| 批次 / 轮次 | Batch / Epoch | 一次喂几张 / 全部过一遍 |
| 过拟合 | Overfitting | 背下训练集答案，遇到新题崩盘 |
| 迁移学习 | Transfer Learning | 从已有权重起步，而非从零训练 |
| 交并比 | IoU | 重叠面积 ÷ 总覆盖面积 |
| 精确率 / 召回率 | Precision / Recall | 摘的对不对 / 真的找到多少 |
| 平均精度 | mAP | 检测领域头号指标 |
| 非极大值抑制 | NMS | 投票去重：分最高当选，高度重合的弃权 |
| 凸包 | Convex Hull | 橡皮筋套住形状围出的区域 |
| 实心度 | Solidity | 面积 ÷ 凸包面积 |
| 距离变换 | Distance Transform | 每像素到最近前景的距离（及坐标） |
| 领域差异 | Domain Gap | 训练数据与真实数据长得不像 → 模型失效 |
| 鸭子类型 | Duck Typing | 有同名属性即可，不需要类型继承 |
| CVAT | Computer Vision Annotation Tool | 开源网页端标注平台 |

---

## 附录 B · 易错点清单（含本人订正）

| # | 易错点 | 正确做法 |
|---|---|---|
| 1 | 把「误差」当正式术语 | 正式术语是**损失（Loss / Loss Function）** |
| 2 | 把学习率写成 `Ir` | 是 **`lr`**（小写 L + r），来自 **l**earning **r**ate；代码里是 `lr0`。小写 l 和 I 极易混 |
| 3 | 文件夹 vs 文件不分 | `process_single_image` 定义在**文件** `yolo_training/calculate_structure_score.py:1290`，该文件属于**文件夹** `yolo_training/` |
| 4 | RGB / BGR 通道顺序 | OpenCV 用 BGR，Gradio/PIL/Matplotlib 用 RGB，**必须 `cvtColor` 转换** |
| 5 | 数据集不洗牌就切分 | 会导致**分布偏移**，必须先 `random.shuffle` |
| 6 | 形态学「先开后闭」还是「先闭后开」 | 本项目是**先闭（填裂缝）后开（去毛刺）** |
| 7 | 忘记类别 ID 的顺序 | `dataset.yaml` 的 `names` 是**有序列表**，下标即 ID，顺序变了推理全错 |
| 8 | 以为两个 IoU 阈值是同一回事 | 评测 IoU（预测 vs 标注）≠ NMS IoU（预测 vs 预测） |
| 9 | 以为本项目自己写了损失函数 | 没有，全部由 **Ultralytics** 内置提供 |
| 10 | 以为图像预处理有去噪 | 没有，项目**刻意不对原图做缩放/降噪**（会影响小目标检测） |

---

## 附录 C · 尚未学习的模块

| 讲次 | 主题 | 关键内容 |
|---|---|---|
| **第 5 讲（下）** | 图像处理流水线后半段 | 清晰度：腐蚀相减取 1 像素细晶界 → 膨胀成分析带 → Sobel 梯度；划痕：骨架化 → PCA 去离群 → Huber 拟合；可视化：NumPy 拼接 + PIL 画中文 |
| **第 6 讲** | 评分算法 | 清晰度 / 划痕 / 假象 / 均匀度如何映射成分数；`backend/scoring_rules.py` 的分段线性映射与材料权重 |
| **第 7 讲** | 推理与部署 | 批量处理、Jetson + TensorRT、Windows 打包、FastAPI/SSE、数据库与 PDF 报告 |
| **第 8 讲** | 知识体系学习路线 | Python 工程结构 → PyTorch → OpenCV → CNN → 训练技巧 的系统学习顺序 |
| **结课作业** | 自己画数据流架构图 | 用自己的话画出从照片输入到分数输出的完整流程 |

---

## 附录 D · 存疑 / 不确定之处

> 以下内容我**未能从代码中确认**，请勿当作定论。

1. **数据集划分是否按「样品」分组**：`prepare_yolo_dataset.py` 是**随机划分**。金相照片常来自同一批同类样品（甚至相邻视场），按图片随机划分可能**乐观高估泛化性能**。更严谨应按「样品/工件」分组划分。项目是否另有脚本处理，未确认。
2. **默认优化器**：`train_yolov8.py` 未显式指定 optimizer，使用 Ultralytics 默认（SGD 或 AdamW 取决于版本），以及默认的学习率调度与早停策略。**未在本仓库中确认具体默认值**。
3. **总分权重口径**：`docs/roadmap.md` 中「最终质量总分公式」标记为 ❌ 未开始（阻塞于专家定义权重）；而 `backend/scoring_rules.py` 的 `MATERIAL_SCORE_WEIGHTS` 已是 clarity 40 / scratch 20 / artifact 20（满分 80）。**说明该权重仍在迭代**，以 `scoring_rules.py` 当前代码为准。
4. **假象评分存在两套口径**：`calculate_structure_score.py` 的 `calculate_artifact_score()` 按**总数**分档（≤2 → 20 分 …）；而 `backend/scoring_rules.py` 的 `compute_artifact_score_from_metrics()` 按**坑洞数 + 明暗比 + 水渍 + 锈迹**多因子计算。二者关系需在运行期确认（脚本内的值随后会被 `enrich_metrics` 重新计算覆盖）。
5. **YOLOv8 与 YOLO26 并存**：文档与脚本口径不一（见 0.3 节），具体线上用的是哪一代权重，需查看部署环境的 `MODEL_PATH`。

---

*笔记整理时间：2026-09-20　|　覆盖第 0~5（上）讲　|　所有代码引用均来自实际读取，未读取权重与数据集文件*
