# PINN-SiC-MOSFET

**基于物理信息神经网络（PINN）的 4H-SiC 功率 MOSFET SPICE 紧凑模型参数提取**

面向 1.2 kV 4H-SiC 垂直 DMOSFET 功率器件，利用物理信息神经网络从 Silvaco TCAD 仿真数据中自动提取 SPICE Level-3 紧凑模型的物理参数，并生成可直接用于 LTspice / ngspice / HSPICE 电路仿真的标准 SPICE `.SUBCKT` / `.MODEL` 文件。整体精度 **MAPE 0.957%**，超越目标（< 3%）3.1 倍。

---

## 项目背景

功率器件紧凑模型是电力电子电路仿真的基石。传统参数提取依赖人工拟合，工作量大、且难以覆盖宽工作范围。本项目将 **PINN（物理信息神经网络）** 引入 SPICE 紧凑模型的参数逆向提取：

- 用 **可微分的 SPICE Level-3 物理模型**（13 个可训练物理参数）承载器件物理；
- 用 **神经网络残差** 修正物理模型与 TCAD 数据之间的偏差；
- 一次训练即可得到**物理可解释、全工作范围可用**的 SPICE 模型。

## 技术路线

```
Silvaco TCAD 仿真数据 (4H-SiC VDMOS)
        │
        ▼
数据预处理 ── 多 V_GS/V_DS/温度 CSV 加载、物理特征初始化
        │
        ▼
PINN 混合模型 ── SPICE Level-3 物理模型 × exp(δ_log) 神经网络残差
        │
        ▼
双阶段联合训练 ── 对数域损失 + 3×亚阈值专项损失 + 物理/平滑/单调约束
        │
        ▼
物理参数提取 ── 13 个参数（VTO、KP、THETA、U0、VMAX、LAMBDA、ETA…）
        │
        ▼
SPICE 模型生成 ── SiC_MOSFET_PINN.lib（.SUBCKT + .MODEL + JFET + 非线性电容）
        │
        ▼
精度验证 ── 分区域/分温度 MAPE、R²、Gummel 对称性测试（GST）
```

## 核心创新

| 创新点 | 说明 |
|--------|------|
| 物理残差架构 | `I_total = I_phys × exp(δ_log)`，乘法修正保证电流恒为正、梯度畅通 |
| 可微分 SPICE Level-3 模型 | 13 个物理参数纯 PyTorch 实现，支持自动微分 |
| 亚阈值专项损失 | 3 倍加权 log-space 损失，显著提升弱反型区精度 |
| 双端对称导通 | 支持 V_DS < 0 反向导通，通过 Gummel 对称性测试（误差 0.67%） |
| 全工作范围覆盖 | V_GS ∈ [0, 20] V，V_DS ∈ [0, 1200] V，T ∈ [298, 448] K |
| 温度依赖内置 | VTH(T) 与 μ(T) 温度模型，覆盖 25–175 °C |

## 精度验证结果

| 指标 | 本工作（PINN-SPICE） | 目标 / 前人 |
|------|---------------------|-------------|
| 整体 MAPE | **0.957%** | < 3%（超越 3.1×） |
| 亚阈值 MAPE | **4.59%** | < 10% |
| 强反型 MAPE | **0.21%** | — |
| R² | **0.99904** | — |
| RMSE | **1.64×10⁻⁶ A** | — |
| GST 对称性 | **0.67%**（通过） | < 1% |

- 78.5% 测试样本误差 < 0.5%，93.5% 样本误差 < 5%
- 分温度：298 K 0.61% / 348 K 0.92% / 398 K 1.59% / 448 K 0.71%
- 对比前次经验模型（仅 V_GS∈[0,12]V 有效、外推发散）：本模型全范围 MAPE 1.62% vs 前次 3.0×10⁸%

## 模型架构

```
输入: [V_GS, V_DS, V_BS, T]  (4 维)
        │
        ▼
PINN 骨干网络 ── 8 层 MLP，256 维隐藏层
        │       Swish + LayerNorm + Dropout(0.05)
        │       残差连接 + 单调性约束
        ▼
δ_log 预测 ── 对数域修正量（最终层零初始化）
        │
        ▼
SPICE Level-3 物理模型 ── 13 个可训练物理参数
        │       I_phys = f(V_GS, V_DS, T)
        ▼
I_total = I_phys × exp(δ_log)  ── 乘法修正
```

### 13 个可训练物理参数（提取值示例）

| 参数 | 含义 | 提取值 | 单位 |
|------|------|--------|------|
| VTO | 零偏阈值电压 | 3.800 | V |
| KP | 跨导参数 | 0.482 | A/V² |
| THETA | 迁移率退化系数 | 0.0261 | V⁻¹ |
| U0 | 低场表面迁移率 | 49.99 | cm²/V·s |
| NFS | 快界面态密度 | 1.97×10¹¹ | cm⁻² |
| GAMMA | 体效应系数 | 1.500 | V^0.5 |
| PHI | 表面势 2φ_p | 0.650 | V |
| VMAX | 载流子饱和速度 | 2.0×10⁷ | m/s |
| KAPPA | 饱和电场因子 | 0.500 | — |
| LAMBDA | 沟道长度调制 | 4.99×10⁻⁴ | V⁻¹ |
| KAPPA_VT | VTH 温度系数 | -0.0060 | V/K |
| ALPHA_MU | 迁移率温度指数 | 2.517 | — |
| ETA | DIBL 系数 | 4.98×10⁻⁴ | — |

### 损失函数

```
L_total = L_data_log + 3.0×L_data_sub + L_data_lin
        + λ_phys×L_physics + λ_smooth×L_smooth + λ_mono×L_mono
```

| 分量 | 公式 | 作用 |
|------|------|------|
| L_data_log | MSE(log₁₀ I_pred, log₁₀ I_true) | 全量程对数精度 |
| L_data_sub | MSE(log I) × mask(I < 1e-6 A) | 亚阈值专项（3× 加权） |
| L_data_lin | MSE((I_pred-I_true)/I_true) × mask(I > 1e-8 A) | 强反型线性域 |
| L_physics | mean(δ_log²) | 物理一致性 |
| L_smooth | mean(d²I/dV_DS²) | I-V 平滑性 |
| L_mono | mean(max(0, ΔV_GS×ΔI)) | V_GS 单调性 |

## 快速开始

### 环境要求

- Python 3.9+，PyTorch（CUDA GPU 推荐）
- Silvaco TCAD（生成仿真数据，可选）
- LTspice / ngspice（验证生成的模型，可选）

### 训练与生成 SPICE 模型

```bash
# 使用 Silvaco TCAD 数据（推荐）
python main.py --data "Silvaco_IV01_csv1.csv"

# 无数据时使用合成数据演示完整流程
python main.py

# 仅验证已保存的模型
python main.py --validate-only

# 仅训练，跳过 SPICE 生成
python main.py --train-only
```

输出：

```
output/
├── SiC_MOSFET_PINN.lib        # SPICE .SUBCKT + Level-3 .MODEL
├── extracted_params.json      # 提取的物理参数
├── training_history.json      # 逐 epoch 训练历史
└── validation/
    └── validation_report.json # 分区域/分温度精度报告
```

### 使用生成的 SPICE 模型

**LTspice**

```spice
.include SiC_MOSFET_PINN.lib
XU1 D G S SiC_MOSFET_PINN
.DC VGS 0 20 0.1
.TEMP 25 75 125 175
```

**ngspice**

```spice
.include SiC_MOSFET_PINN.lib
XM1 D G S SiC_MOSFET_PINN
.dc VGS 0 20 0.1
.temp 25 75 125 175
```

## 项目结构

```
PINN-SiC-MOSFET/
├── main.py                      # 完整流水线入口（数据→训练→提取→生成→验证）
├── improved_VDMOS.in            # Silvaco TCAD 4H-SiC VDMOS 仿真脚本
├── Silvaco_IV01_csv1.csv        # 示例 TCAD 仿真数据
├── AI_Prompt.md                 # 项目方案设计 Prompt（文献调研→路线图→方案）
├── data_processing/
│   └── load_tcad.py             # TCAD 数据加载、物理特征提取、数据集构建
├── pinn_model/
│   ├── level3_physics.py        # 可微分 SPICE Level-3 物理模型（13 参数）
│   ├── pinn_network.py          # PINN 骨干网络与混合模型
│   └── train.py                 # 双阶段训练器（AdamW + CosineAnnealing）
├── spice_gen/
│   └── generate_spice.py        # SPICE .lib 生成器
├── validation/
│   └── validate.py              # 分区域/分温度验证 + GST 对称性测试
├── IV_LTspice/                  # LTspice 验证电路（I-V / CISS / COSS / CRSS）
├── IV_MATLAB/                   # MATLAB 紧凑模型拟合（10 参数 I-V 模型）
├── IV_Silvaco/
│   └── VDMOS.txt                # 早期 Silvaco 仿真脚本
└── output_final/                # 最终交付物
    ├── SiC_MOSFET_PINN.lib
    ├── extracted_params.json
    ├── 实验报告.md
    └── validation/validation_report.json
```

## 关键问题与解决

| 问题 | 解决方案 |
|------|----------|
| 训练初始输出 NaN | 加法修正 → 乘法修正 `I_phys × exp(δ_log)` |
| 物理参数崩塌 | 去除 Stage-1 预热，改为联合训练 + 分离学习率 |
| 周期性 MAPE 尖峰 | CosineAnnealingWarmRestarts → CosineAnnealingLR 单调衰减 |
| 亚阈值精度差（13.4%） | 新增 3× 亚阈值专项 log-space 损失 |
| V_DS=0 处电流泄漏 | 缩窄饱和区过渡带宽度 |

## 许可

本项目仅供学术研究与交流使用。

---

*Author: Charles (Zonkit) · 4H-SiC Power MOSFET Modeling Project*
