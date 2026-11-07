# 公开源域基准与本地女足球队评价

整理日期：2026-11-07。此版本整理既有分析文件，未在上传过程中重新训练或更改研究结果。

## 从哪里开始

公开数据的**主观＋客观合并主表**为：

- [source_athlete_day_preprocessed.csv](source/verified_outputs/01_source_preprocessing_v1/source_athlete_day_preprocessed.csv)：可用Excel/WPS查看。
- [source_athlete_day_preprocessed.parquet](source/verified_outputs/01_source_preprocessing_v1/source_athlete_day_preprocessed.parquet)：供代码读取，与CSV为同一层级数据。
- [source_athlete_day_raw_union.csv](source/verified_outputs/01_source_preprocessing_v1/source_athlete_day_raw_union.csv)：较宽的原始来源主键并集，供资格审计使用。

| 数据层级 | 运动员 | 行数 | 字段 | 阳性标签 | 用途 |
|---|---:|---:|---:|---:|---|
| 来源合并并集 | 78 | 21,729 | 65 | 见原审计表 | 保留来源与资格审计 |
| 50人合并基础主表 | 50 | 19,689 | 65 | 156 | 主观、训练、GPS日级合并 |
| 六个历史窗口的共同分析队列 | 50 | 19,613 | 75项输入＋3项标识/结局 | 149 | injury版模型比较与冻结 |

合并基础主表约6.77 MB（CSV）或1.38 MB（Parquet）。合并是从会话与高频监测资料提取日级指标，不是对原始轨迹做无损压缩。高频GPS原文件不在本仓库重复存储。

## 结局、窗口和版本

当前主线使用原始日级 `injury` 二分类标签。源域来自 `injury_report_t`，表示运动员当日是否提交至少一次伤病通知；历史输入仅使用目标日前的自然日，区间为 `[t-W, t-1]`。窗口为2、4、8、16、21、32天，六个窗口使用同一分析队列。149个阳性日是通知日数量，不是临床新发损伤次数。

合并主表保留 `y_t1`、`target_date` 等历史审计字段。当前 `03n` 明确把 `injury_report_t` 复制为 `injury`；这些旧字段不进入预测变量。旧的137阳性、146阳性`t+1`链及Emon的8,582行原始版本未混入当前分析。

当前保留的源域模型是**开发阶段选定并冻结的8日LightGBM候选**。开发比较中的AP为0.042973，属于选择候选时的开发结果。联合模型—窗口选择程序的嵌套评价另见 `source/nested_48_reconciliation/RECONCILED_REPORT.json`，其AP为0.012877、ROC-AUC为0.573424。两种结果评价的对象不同，冻结候选并不等于联合选择程序每折都选中同一个模型。

## 目录

```text
soccer_injury_thesis/
  source/                    公开数据处理、建模、源域结果与模型
    data/subjective/         复现合并所需的公开主观输入
    verified_outputs/        合并主表、GPS日级缓存、通知日表
    outputs/injury_history_lopo_v1/
                             六窗口输入、模型比较、调参、校准、SHAP
    nested_48_reconciliation/ 论文采用的48配置复核结果
    cache/                   4800折预测文件的无损分卷归档及成员哈希
  local/数据处理方案/        本地审计、重建、对齐与LOPO更新代码
  tools/                     快速核查、缓存恢复、冻结模型评分
  docs/                      程序输入输出、数据字典、版本说明
  licenses/                  上游代码许可
  BUILD_PROVENANCE.json       原文件与发布副本的哈希及路径调整记录
```

`verified_outputs`是既有输出目录名称。主客观合并的实际生成程序是 `01_soccermon_preprocessing.py`。文件编号反映开发过程；当前injury版特征工程位于 `03n_injury_history_lopo.py --stage features`，不沿用旧`02`的`t+1`标签设定。

## 环境与核查

运行环境记录为Python 3.13.13；主要依赖见 [requirements.txt](requirements.txt)。这份代码包含`fcntl`锁及多进程，建议使用macOS/Linux。Joblib模型应仅加载可信来源，并使用匹配的scikit-learn/LightGBM环境。

```bash
cd soccer_injury_thesis
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python tools/verify_release.py
python tools/verify_release.py --full
```

普通核查检查文件哈希、归档、代码语法及CSV主表；`--full`增加Parquet规模、特征契约、六窗口标签一致性、已存预测指标及冻结模型加载检查，不重新训练。

4800折预测缓存以六个`.partNN`分卷传输。恢复脚本按清单拼接、核对整档哈希后读取，不改变原始CSV内容。

## 运行顺序

详细输入输出见 [PIPELINE.md](docs/PIPELINE.md)。仓库已经包含合并表、历史特征、既有模型与结果，查看资料不需要重新训练。重跑会向相同输出位置写文件，建议在独立克隆或另存副本中运行。

```bash
# 恢复4800折既有预测缓存，便于复核及需要缓存的后续脚本。
python tools/restore_benchmark_cache.py

cd source
# 可选：从已包含的公开主观原始表和GPS日级缓存重建合并表。
python verified_07_source_injury_reportday_t1.py
python 01_soccermon_preprocessing.py

# 当前injury历史特征工程、EDA。
python 03n_injury_history_lopo.py --stage features
python 03p_injury_eda.py

# 八模型 × 六窗口 × control/训练折内ROS；完整计算较耗时。
python 03n_injury_history_lopo.py --stage benchmark --workers 4

# 调参基础模块及初期候选；前三个开发配置的分组嵌套调参。
python 03o_injury_tuning_and_freeze.py --workers 4
python 03q_top3_injury_tuning.py

# 联合模型—窗口选择；论文采用的48配置结果由复核脚本生成。
python 03r_injury_nested_joint_selection.py --workers 4
python reconcile_nested.py

# 固定候选校准、SHAP、图表、冻结供本地使用的原始概率管线。
python 04c_injury_nested_probability_calibration.py
python 04d_injury_shap.py
python 04e_injury_calibrated_model_export.py
python 05b_injury_chapter3_assets.py
python 03s_freeze_injury_source_candidate.py
```

`reconcile_nested.py`使用既存完整内层分数统一48个control配置，复现每折选中配置的3个内层评分，再重拟合50个外层模型。上传结果来自这一复核，未把它写成重新执行全部7200个内层拟合。上游GPS逐点重算脚本 `verified_04a_source_gps_daily.py` 一并保留；运行它需自行下载并按`data/2020`、`data/2021`放置高频数据。本仓库包含01实际读取的GPS日表，因而可从日级缓存开始复现合并。

## 本地队伍代码与数据访问

本地代码按第四、五章实际流程保留，见 [local/README.md](local/README.md)。原始健康表、姓名、逐日伤病标签及个人级预测结果**未公开上传**。它们只在经作者确认的私有仓库保存；公开仓库的 `.gitignore` 也排除了本地资料目录。

源域使用运动员自主通知，本地使用队医记录，两者按各自记录定义解释。当前本地更新是截距校准、截距—斜率校准和固定特征/超参数的本地LightGBM重新拟合，不是接续源域树模型进行增量训练。第六章汇总第三至五章，不另建一套模型。

## 数据来源与许可

SoccerMon数据来自[官方数据页面](https://datasets.simula.no/soccermon/)和[Zenodo](https://zenodo.org/records/10033832)，采用[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)许可。本目录中的日级聚合、队列筛选和历史窗口特征是基于原数据制作的派生数据，改变过程见对应脚本、运行清单和数据字典。

请引用：Midoglu C, et al. **A large-scale multivariate soccer athlete health, performance, and position monitoring dataset.** *Scientific Data*. 2024;11:553. DOI: [10.1038/s41597-024-03386-x](https://doi.org/10.1038/s41597-024-03386-x)。

代码研究背景参照Md Mohaiminul Islam Emon的硕士论文及[Simula PMSYS项目](https://github.com/simula/pmsys/tree/main/md-mohaiminul-islam-emon)。上游MIT许可保留于 [licenses/PMSYS_LICENSE.txt](licenses/PMSYS_LICENSE.txt)。本地队伍数据不随SoccerMon的公开数据许可发布。
