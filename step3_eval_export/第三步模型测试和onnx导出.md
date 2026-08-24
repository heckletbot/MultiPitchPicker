# 第三步：模型测试和 ONNX 导出

本步只做两件事：用**已经有的**测试集评模型，然后导出 ONNX。

测试集不在本步生成。第一步 09 已经写出 `data/synth/test.jsonl`（默认是留出音色，训练没见过）。本步只读这份清单扫阈值、写报告。没有 `test.jsonl` 就回去跑第一步，不要在这里重造。

本步输入是第二步的权重（`data/runs/{mix,base}/best.pt`，以及可选的 `step2_train/work/selected.json`）。  
本步输出：


| 产物                      | 路径                                                            |
| ----------------------- | ------------------------------------------------------------- |
| 完整测评报告                  | `step3_eval_export/work/test_report.json`（同时写到 ckpt 目录）       |
| 推荐阈值 τ                  | `step3_eval_export/work/selected.json`（同步写回 step2 的 selected） |
| 图内 CQT 偏置               | `data/runs/torch_cqt_bias.pt`                                 |
| ONNX                    | `export/model.onnx`                                           |
| 端侧元数据                   | `export/meta.json`                                            |
| 真录健康度（有 `data/real/` 时） | `step3_eval_export/work/real_health.json`                     |


---

## 依赖

```bash
pip install -r step3_eval_export/requirements.txt
```

先确认：

- 第一步：`data/synth/test.jsonl` 存在（以及对应 wav / CQT 缓存）
- 第二步：`data/runs/mix/best.pt` 或 `data/runs/base/best.pt` 存在。有 `step2_train/work/selected.json` 会优先用里面的 ckpt / lookahead

根目录 `config.py` 的 `MIDI_LO/HI` 必须与训练时一致。

---

## 运行

默认从评测跑到导出（有 `data/real/` 时末尾加健康度）：

```bash
python step3_eval_export/scripts/run_pipeline.py
```

流水线四步，后一步依赖前一步产物：


| 步   | 脚本                    | 输入                                  | 输出                                     |
| --- | --------------------- | ----------------------------------- | -------------------------------------- |
| 评测  | `01_eval_testset.py`  | 第二步权重 + 第一步 `data/synth/test.jsonl` | `selected.json`（最佳 τ）                  |
| 校准  | `03_calibrate_cqt.py` | `data/synth/validation.jsonl`       | `torch_cqt_bias.pt`                    |
| 导出  | `04_export_onnx.py`   | 权重 + bias + τ                       | `export/model.onnx`、`export/meta.json` |
| 健康度 | `02_real_health.py`   | `data/real/` + 已导出 ONNX             | `real_health.json`；无真录则跳过              |


可选参数（可组合；只跑一条命令）：


| 参数                 | 作用                            |
| ------------------ | ----------------------------- |
| `--from calibrate` | 从校准起跑，跳过评测                    |
| `--from export`    | 从导出起跑，跳过评测与校准                 |
| `--from health`    | 只跑健康度                         |
| `--reuse-selected` | 评测步不重算 τ，沿用已有 `selected.json` |
| `--skip-health`    | 跳过健康度                         |


---

## 分步

```
step2 的 ckpt（mix 优先，否则 base）
  └─ 01 读 data/synth/test.jsonl 扫 τ，写 test_report.json + selected.json
        └─ 03 用 validation 校准图内 CQT → torch_cqt_bias.pt
             └─ 04 export_onnx.py → export/model.onnx + meta.json
                  └─ 02（可选）对 data/real 滑窗转录，统计密度 / 调内占比
```

```bash
python step3_eval_export/scripts/01_eval_testset.py
python step3_eval_export/scripts/03_calibrate_cqt.py
python step3_eval_export/scripts/04_export_onnx.py
python step3_eval_export/scripts/02_real_health.py          # 需要已导出的 ONNX
```

超参在 `[config_eval.json](config_eval.json)`。`meta.json` 字段对齐 `MultiPitchPicker-2/deploy/assets_bpm/meta.json`（sr / hop / fps / midi_lo / num_pitches / thresh_default / min_gap_frames / peak_floor）。

`thresh_default` 用本步 01 在第一步测试集上扫出的 `best_thresh`，不是 checkpoint 里 val 的 τ。

低延迟变体：导出前设 `MP_CQT_GAMMA=2`，且 ckpt 是 `--lookahead` 训的。bias 必须同 γ 校准。离线客户端不用开。

---

## 怎么看测试集指标

`01_eval_testset.py` 读第一步的合成 **test**（默认留出音色），写出 `test_report.json` 和 `selected.json`。匹配规则：预测音与标注音 **音高相同**，起音时间差在 **±50 ms** 内算一对。

看这些字段：


| 字段                     | 含义                                           |
| ---------------------- | -------------------------------------------- |
| `n`                    | 测试片段条数                                       |
| `best_thresh`（τ）       | 在扫描范围内 F1 最高的解码门槛。导出和客户端用这个，不要照抄马林巴的 0.8     |
| `precision`（P）         | 报对的音 / 模型报出的全部音。低说明假音多                       |
| `recall`（R）            | 报对的音 / 标注里应有的音。低说明漏音多                        |
| `f1`                   | P 和 R 的调和平均                                  |
| `velocity_mae`         | 起音处力度误差（0–127）。越小越好                          |
| `warn_f1_below_accept` | F1 低于 `config_eval.json` 警戒线（默认 0.98）时为 true |


对照 `bpm_mix`（合成留出槌）：P 99.5 / R 99.3 / F1 99.4，力度 MAE 约 7，τ 约 0.8。打击键盘可把 F1 ≥ 0.98 当作是否导出的参考；钢琴/吉他往往略低。数字明显差时先查第一步的单音库、切分和音域，不要导出。

真录没有乐谱：02 做无监督健康度（音符密度、音域、调内占比）。马林巴独奏调内常 >95%；转调乐曲会低一些。

---

## 脚本


| 脚本                    | 作用                                                      |
| --------------------- | ------------------------------------------------------- |
| `01_eval_testset.py`  | 读第一步 `data/synth/test.jsonl` 扫 τ，写完整 `test_report.json` |
| `02_real_health.py`   | 真录滑窗转录统计（解码 = `infer_onnx`，需已导出）                        |
| `03_calibrate_cqt.py` | 调 `torch_cqt.py --save-bias`                            |
| `04_export_onnx.py`   | 调 `export_onnx.py`，meta 的 τ 用测评结果                       |
| `run_pipeline.py`     | 评测 → 校准 → 导出（可选健康度）                                     |


权重仍在 `data/runs/`，ONNX 在仓库根目录 `export/`，中间报告在 `step3_eval_export/work/`。

完成后下一步见 [第四步：模型调用和实战测试方法](../step4_api/第四步模型调用和实战测试方法.md)。