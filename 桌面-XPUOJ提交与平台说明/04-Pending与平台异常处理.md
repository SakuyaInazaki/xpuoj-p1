# 04 Pending 与平台异常处理

## 1. 正常情况

- 提交后状态：`Pending -> Accepted/WrongAnswer/TimeLimitExceeded`。
- 简单改动一般 1-4 分钟出结果。
- `submit.py --poll` 会打印每次轮询到的状态 JSON。

## 2. 偶发长时间 Pending（20-30 分钟）

历史会话中多次出现：

```text
117032 / v177：Pending 很久，最终 TimeLimitExceeded
117077-117078 / v199 与对照：Pending 很久后正常 Accepted
117083-117084 / down w16 与对照：Pending 很久后正常 Accepted
```

### 遇到 Pending 怎么办

1. **不要立即重新提交同一文件。**
2. 用独立工具查详情：

```bash
python "桌面-XPUOJ提交与平台说明/tools/xpuoj_status.py" 117xxx
```

3. Pending 时 `progress` 可能为 `null`，meta.status 可能仍是 `Pending`，这是正常的。
4. 如果本地轮询超时，不要以为评测失败。平台进程通常还在跑。

### 判断是否真卡住

- 观察其他简单提交是否也 Pending：
  - 如果其他新提交正常出结果，可能是该提交本身很慢（编译慢/TLE）。
  - 如果所有新提交都 Pending，可能是平台排队。
- 总时间限制以题目为准（历史某题为 500 秒）；TLE 通常会在限制附近返回。超过 30 分钟仍 Pending 也可继续等或改做本地工作。

## 3. 工具层 300 秒超时 vs 平台评测超时

执行环境给 bash 命令的默认超时可能是 300 秒：

```text
Your command timed out after 300 seconds ...
```

这**不代表平台评测失败**。提交已经发生，只是本地 `--poll` 被杀了。

处理：

```bash
# 不要再 submit，只查状态
python "桌面-XPUOJ提交与平台说明/tools/xpuoj_status.py" <submissionId> --cases
```

## 4. 提交成功但日志没有终态

如果提交时没用 `--poll`：

```text
submitted submissionId=117310
```

日志里只有这一行。最终状态必须从 API 补：

```bash
python "桌面-XPUOJ提交与平台说明/tools/xpuoj_status.py" 117310 --cases
```

## 5. 查最近提交的接口限制

`querySubmissions` 的 `takeCount` 写大也没用，基本只回最近约 10 条。
旧 submission 用 `getSubmissionDetail` 查。

## 6. HTTP / 网络错误

- `requests` 超时或 5xx：稍等重试查询，不要重复提交。
- `submit rejected` / 非 200：打印响应原文，通常是 code/language 或 JSON 结构问题。
- 登录失败：检查 `.secrets/xpuoj.json` 路径和权限。

## 7. 平台没有可靠 cancel 接口（历史观察）

详情 meta 里可能有 `permissionCancel` 字段，但本会话没有验证过对应 API。
不要依赖 cancel；Pending 提交自然结束即可。

## 8. 推荐工作方式

```text
提交 -> 记录 submissionId -> 不阻塞继续做本地/其他工作 -> 定期查 status
```

不要同步死等 Pending。
