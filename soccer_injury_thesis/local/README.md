# 本地目标域代码

此公开目录只含代码。姓名、原始问卷、队医伤病表、GNSS文件、本地日级合并表、逐人预测和各类个人级审计结果不在本仓库公开。

| 顺序 | 程序 | 输入 | 输出目录 |
|---|---|---|---|
| 1 | `04_chapter4_local_target_audit.py` | 原始问卷、队医表、主客观资料 | `13_目标域原始审计` |
| 2 | `05_target_domain_daily_rebuild.py` | 相同原始资料及审计 | `14_目标域日级重建` |
| 3 | `15_source_candidate_target_alignment.py` | 日级表、客观会话文件、源域冻结模型 | `15_源域候选迁移对齐` |
| 4 | `16_target_domain_update_lopo.py` | 对齐后的8日特征、同日injury及源域冻结模型 | `16_目标域模型更新_LOPO` |

代码只调整了目录定位方式，计算逻辑与本机原脚本一致。默认数据根目录为本目录，可通过`LOCAL_DATA_ROOT`指定受控目录；源域根目录默认是同级`../source`，也可通过`SOCCERMON_ROOT`指定。

```bash
export LOCAL_DATA_ROOT="/path/to/private-local-data"
export SOCCERMON_ROOT="/path/to/work/soccer_injury_thesis/source"
python 数据处理方案/04_chapter4_local_target_audit.py
python 数据处理方案/05_target_domain_daily_rebuild.py
python 数据处理方案/15_source_candidate_target_alignment.py
python 数据处理方案/16_target_domain_update_lopo.py
```

若已经持有对齐后的`15_源域候选迁移对齐/target_domain_history_8d_aligned.csv`，可直接运行第4项，重建本地24人LOPO更新分析。第3项会从原始GNSS文件重算跑动指标，因此仅有日级面板而没有原始GNSS文件时，应使用受控保存的对齐结果，不把缺少的客观信息改写为零。

模型更新包括源域管线直接评分、截距更新、截距—斜率更新，以及固定源域特征规格和超参数的本地重新拟合。代码中不另做本地超参数海选。源域通知标签与本地队医记录保留各自含义。
