# 程序—输入—处理—输出

## 当前正式主线

以下源域相对路径以`source/`为起点。

| 阶段 | 代码 | 实际处理 | 主要输出 |
|---|---|---|---|
| 通知日审计 | `verified_07_source_injury_reportday_t1.py` | 读取公开伤病CSV、完全重复审计、按人日归并；01使用其原始报告日表 | `verified_outputs/07_source_injury_reportday_t1_v1/04_injury_report_day.csv` |
| GPS处理 | `verified_04a_source_gps_daily.py` | 高速饱和/异常点质量控制、有效时间间隔积分及日级聚合 | `verified_outputs/04_source_gps/source_gps_daily.csv` |
| 主客观合并 | `01_soccermon_preprocessing.py` | 会话、问卷、负荷、表现、疾病、GPS及通知日合并；来源资格审计 | `verified_outputs/01_source_preprocessing_v1/source_athlete_day_preprocessed.*` |
| injury特征工程 | `03n_injury_history_lopo.py --stage features` | 人内自然日历、先滞后1日、六个历史窗口、共同队列 | `outputs/injury_history_lopo_v1/history_*d.parquet` |
| EDA | `03p_injury_eda.py` | 样本与结局分布、月度分布、来源可得性 | `eda_*.csv/.json`及图 |
| 八模型比较 | `03n_injury_history_lopo.py --stage benchmark` | LR、GaussianNB、RBF-SVM、RF、XGBoost、LightGBM、CatBoost、MLP；6窗口；control及训练折内ROS；50人LOPO | `benchmark_summary.csv`及4800折预测 |
| 调参基础 | `03o_injury_tuning_and_freeze.py` | 分组内层评分、训练内插补/缩放/重采样、候选整理 | `tuning/`、初期模型包及预测 |
| 前三配置调参 | `03q_top3_injury_tuning.py` | 对8d-LGBM-control、32d-XGB-control、21d-LGBM-ROS三配置各做嵌套搜索 | `top3_tuning/` |
| 联合选择 | `03r_injury_nested_joint_selection.py` | 50外层LOPO、3内层运动员组折、模型×窗口选择 | `nested_joint_selection/` |
| 48配置统一复核 | `reconcile_nested.py` | 从完整内层存档筛出48个control配置，复现选中配置内层结果、重拟合50外层 | `nested_48_reconciliation/RECONCILED_REPORT.json`及预测 |
| 固定候选校准 | `04c_injury_nested_probability_calibration.py` | 训练运动员内交叉拟合概率校准 | 校准OOF预测、校准报告 |
| SHAP | `04d_injury_shap.py` | 固定源域候选真实样本的模型贡献 | SHAP数组、特征贡献表、图及样本索引 |
| 校准包导出 | `04e_injury_calibrated_model_export.py` | 导出单独的带校准器模型包 | `selected_injury_model_calibrated.joblib` |
| 论文图表 | `05b_injury_chapter3_assets.py` | 从真实预测与报告绘制PR/ROC/校准等 | `figures/`、素材清单 |
| 本地用冻结模型 | `03s_freeze_injury_source_candidate.py` | 全源域重拟合开发选定的8日LGBM，保存插补与75字段顺序 | `frozen_source_candidate/` |
| 本地重建与直接评价 | `local/数据处理方案/04...`、`05...`、`15...` | 原始资料审计、人日重建、共有变量映射、历史输入及冻结管线评分 | 私有本地资料目录的13、14、15阶段 |
| 本地更新评价 | `local/数据处理方案/16_target_domain_update_lopo.py` | 24人LOPO；直接应用、两种校准更新、固定参数从头拟合 | 私有目录的16阶段 |

前三配置含两个LightGBM配置，属于“排名前三的配置”，不是三个不同模型家族。

## 基础模型参数与搜索设置

八模型默认参数由`source/config_soccermon.py`的`BASELINE_PARAMS`定义；缩放与插补由同文件`make_pipeline`定义。调参空间见`TUNING_SPACES`及`top3_tuning/top3_search_spec.json`。正式参数以这些机器可读文件为准，README不另抄一份可能过期的参数值。

模型比较与调参的AP来自合并外层预测。各折只有一种结局时，单折ROC-AUC不适用；合并指标与运动员聚类Bootstrap按脚本处理。重采样只改变训练折，测试运动员记录保持原样。

## 复核版本与历史编号

`03r`存档包含48与96候选范围的开发历史。本发布保留完整逐折内层分数作为复核输入，不发布旧联合汇总充当正式结果；论文采用`nested_48_reconciliation/RECONCILED_REPORT.json`。若从零重新运行全部联合选择，应使用独立输出副本，与存档复核分开记录。

原项目的`01_soccermon_data_preprocessing.py`是旧137阳性`t+1`面板数值/QC程序；旧`02_soccermon_feature_engineering.py`使用旧`t+1`定义；旧`03_soccermon_eda.py`、`03a`、`03b`属于早期链。本发布以当前实际读取的`01_soccermon_preprocessing.py → 03n及后续`为主线，避免按文件编号误运行旧结局。

第六章汇总这些实证结果，没有独立的“第六章机器学习模型”。旧Word草稿生成器含硬编码历史数值，未作为分析程序打包。
