# 数据层级与字段含义

完整65字段映射见`source/verified_outputs/01_source_preprocessing_v1/SOURCE_FIELD_MAPPING.csv`；缺失情况见同目录`SOURCE_MISSINGNESS.csv`。

| 字段组 | 代表字段 | 含义/单位 |
|---|---|---|
| 主键 | `player_name`, `date` | 公开数据运动员代号、自然日；不作为预测特征 |
| 训练会话 | `session_rpe_mean`, `session_srpe_sum_au`, `session_duration_sum_min` | RPE会话均值、RPE×时长的日内合计（AU）、时长合计（min） |
| 主观健康 | `fatigue_raw`, `mood_raw`, `readiness_raw`, `sleep_duration_raw`等 | 来源量表原值；具体方向由字段映射与代码规定 |
| 方向统一指标 | `fatigue_burden`, `poor_sleep_burden`, `soreness_burden`, `stress_burden` | 源域1—5评分转换`(5-raw)/4`；本地Hooper1—7评分转换`(raw-1)/6` |
| GPS跑动 | `total_distance_m`, `distance_above_15p0_km_h_m`, `gps_max_speed_m_s` | 总距离（m）、速度高于15 km/h的距离（m）、最大速度（m/s） |
| 导出负荷 | `daily_load_au`, `weekly_load_au`, `atl_exported_au`等 | 来源系统导出值；当前历史输入只选规定的14项基础字段 |
| 数据来源标志 | `has_training_record`, `has_wellness_record`, `has_gps_record` | 来源是否有记录；没有记录不自动等于负荷零 |
| 当前源域结局 | `injury_report_t` → 历史表中的`injury` | 当日是否有至少一次自主伤病通知，二分类 |
| 旧分支字段 | `y_t1`, `target_date`, `outcome_boundary_flag` | 合并表保留的旧审计信息，不用于当前injury模型 |

历史表的14项基础变量各生成均值、标准差（ddof=0）、最大值、有效值覆盖率及前一自然日值，共70项；另有3项来源报告率、历史有效观测日数及自然日覆盖度，合计75项。75项顺序保存在`frozen_source_candidate/SELECTED_FEATURES.json`。

均值、标准差、最大值使用历史窗口内实际可用值，缺失不前向填充。前一日值仅指前一自然日，不是向前寻找最近一次有效报告。模型阶段的中位数插补在相应训练数据上拟合。

原始合并主表不等于模型矩阵。19689行基础表经过统一历史纳入条件后得到19613行；被排除的76行及7个阳性见`excluded_no_32day_history.csv`。不同文件层级的阳性数应与其纳入条件一起读取。
