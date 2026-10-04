# XPUOJ P1 MegaMoE 交接（本会话全量版：submission 116944-117310）

> 上一份：`../handoff-session-20260818-final4/`（截至 116897，base v159/116882）。
> 本文件夹汇总从接手 final4 之后，本 agent 从 submission 116944 到 117310 的全部
> 尝试、晋升、踩坑、当前代码状态与可直接接手的信息。
> 本会话中间草稿：`../handoff-session-20260818-final5/`。

## 10 秒结论

- 比赛：XPUOJ contestId=13, problemOrder=1, language=`triton-dist`，账号 `dpsk-test`。
- scoreboard best：**116961，raw 76.67，扣罚后 66.67**（case5 tb=264.994 异常，不是真实性能）。
- **实际性能 base / 当前 `p1/kernel.py`：117300（v233），timeUsed 43266；复测 117304=43325。**
- 从 final4 的 v159 到当前 base，主要晋升：
  - final gather 2D tiling（最大项）；
  - case2 token gather BM64/BH256；
  - case2 token quant BK64；
  - case2 SwiGLU BN64；
  - route E32 BM64、E96 BM64/BK128/w4、E32/E64 BK128；
  - fused gateup GROUP_M=32。
- submissionCount：906；扣罚 `min((attempt-100)*0.1,10)` 已到上限 10。

## 阅读顺序

1. `01-current-state.md` —— 当前文件、SHA、逐点结果、backups、检查清单
2. `02-competition-and-platform.md` —— 比赛/题目/评分/API/噪声/Pending 现象
3. `03-architecture.md` —— 当前 v233 代码路径与关键函数行号
4. `04-session-timeline.md` —— 本会话全部提交表与晋升/关闭结论
5. `05-pitfalls-and-sandbox.md` —— 本会话新踩坑 + final4 继承坑
6. `06-candidates-and-next-steps.md` —— 候选文件、当前差距、下一步优先级
7. `submissions_raw_116944-117310.json` —— 172 次提交 meta + 逐点 tk/tb/score
8. `id_to_logfile.json` —— submissionId 到 logs 文件名的映射
9. `kernel_117300_current.py` / `kernel_117234_backup.py` —— 当前/上一 base 源码副本

## 接手检查清单

```bash
cd /home/sakimi26/xpuoj-p1
python scripts/best_score.py
# scoreboard 应输出 totalScore 66.67, submissionId 116961

sha256sum p1/kernel.py p1/kernel_117300_backup.py p1/kernel_117234_backup.py
python -m py_compile p1/kernel.py p1/kernel_117300_backup.py
```

关键 SHA-256：

```text
14329d914f56144d2ab181eee096e96b61f6862ced5af23af8c7e9e967f1a8c5  p1/kernel.py (=117300, 当前实际 base)
14329d914f56144d2ab181eee096e96b61f6862ced5af23af8c7e9e967f1a8c5  p1/kernel_117300_backup.py
ef9eca54091d4b1e595a41b2afc6e84bddc9a6d62b9d7f9320faef448c7a9640  p1/kernel_117234_backup.py (=117234, gather BH256 中间 base)
76dddada65c070b978db4c33444510642eb3e7b7a34529613caf569a4c35c750  p1/kernel_117218_backup.py (=117218, gather BM64 中间 base)
```

提交新实验：

```bash
cd /home/sakimi26/xpuoj-p1
python -m py_compile p1/<candidate>.py
python scripts/submit.py p1/<candidate>.py --poll --interval 10 --timeout 1800 2>&1 | tee logs/submit_<candidate>.log
```

> 注意：平台会偶发 20-30 分钟 Pending；命令被工具层 300s 杀掉不代表评测失败，
> 用 `python scripts/best_score.py` 或 `Client.get_detail()` 查，不要重复提交同一文件。
