# 02 API 接口与结果解析

API 基础地址：

```text
https://sd629vuj4f7uh2cscrbe0.apigateway-cn-beijing.volceapi.com/api/
```

凭据文件：

```text
/home/sakimi26/xpuoj-p1/.secrets/xpuoj.json
```

凭据是账号密码（当前 `dpsk-test`），**不要打印、不要写进提交文件、不要提交**。

## 通用请求格式

```python
import requests, json
API = "https://sd629vuj4f7uh2cscrbe0.apigateway-cn-beijing.volceapi.com/api/"
s = requests.Session()
creds = json.load(open("/home/sakimi26/xpuoj-p1/.secrets/xpuoj.json"))
r = s.post(API + "auth/login", json=creds, timeout=30)
token = r.json()["token"]
s.headers.update({"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
```

之后所有接口都是 `POST`，JSON body。

## 1. 登录

```http
POST /api/auth/login
```

Body 即凭据文件内容。成功响应含 `token`、`username`。

## 2. 提交代码

```http
POST /api/contest/play/submit
```

Body（实测）：

```json
{
  "contestId": 13,
  "problemOrder": 1,
  "content": {
    "language": "<题目要求的 language>"
    "code": "<完整 Python 文件内容>",
    "compileAndRunOptions": {}
  }
}
```

成功响应示例：

```json
{ "submissionId": 117300 }
```

其他题只需要改 `problemOrder` 和 `language`；`contestId` 一般仍为 13。
`compileAndRunOptions` 无特殊需求时传 `{}`。

## 3. 最近提交列表

```http
POST /api/contest/play/querySubmissions
```

Body：

```json
{
  "contestId": 13,
  "problemOrder": 1,
  "takeCount": 10,
  "locale": "zh_CN"
}
```

响应：

```json
{ "submissions": [ { "id": 117300, "status": "Accepted", "displayScore": 73.92, "timeUsed": 43266, ... } ] }
```

**实测坑：`takeCount` 写 50/100/500 也只回最近约 10 条。**
需要查旧 submission 必须用 `getSubmissionDetail`。

## 4. Scoreboard（当前账号）

```http
POST /api/contest/play/getContestScoreboardMe
```

Body：

```json
{ "contestId": 13 }
```

响应关键结构：

```json
{
  "entry": {
    "totalScore": 66.67,
    "problemScores": [
      {
        "score": 66.67,
        "submissionId": "<当前最佳submissionId>",
        "submissionCount": 906,
        "firstAcceptedAt": "..."
      }
    ]
  }
}
```

`submissionCount` 是该题当前账号的总提交次数，扣罚由它计算。

## 5. 提交详情（最重要的解析入口）

```http
POST /api/submission/getSubmissionDetail
```

Body：

```json
{ "submissionId": "117300", "locale": "zh_CN" }
```

响应结构：

```json
{
  "meta": {
    "id": 117300,
    "status": "Accepted",
    "score": 100,
    "displayScore": 73.92,
    "timeUsed": 43266,
    "memoryUsed": 2330000,
    "problem": { "displayId": 40001, ... },
    ...
  },
  "progress": {
    "status": "Accepted",
    "score": 100,
    "displayScore": 73.92,
    "compile": { "success": true, "message": "" },
    "subtasks": [...],
    "testcaseResult": {
      "<testcaseHash>": {
        "input": "1\n",
        "status": "Accepted",
        "displayScore": 75,
        "score": 100,
        "userOutput": "OJCHAL v1 ...\nOJRESULT v1 <hash> <base64json>",
        "userError": "...",
        "memory": ...,
        "checkerMessage": "..."
      },
      ...
    }
  }
}
```

注意：
- `progress` 在 Pending 时可能为 `null`。
- `testcaseResult` 的 key 是 testcase hash，不是 case 序号；用 `input` 字段识别 case。
- `input` 是字符串，末尾带 `\n`，比较时 `v["input"].strip()`。

## 6. 解析逐点 tk / tb / score

`userOutput` 里有：

```text
OJRESULT v1 <hash> <base64json>
```

base64 解码后是：

```json
{
  "schema_version": 2,
  "tk_time_ms": 10.183,
  "tb_time_ms": 28.871,
  "th_time_ms": 0.0,
  "pass": true
}
```

- `tk_time_ms`：该测试点 kernel 时间（毫秒）。
- `tb_time_ms`：该测试点 baseline/对比时间（毫秒）。
- `displayScore`：单点分数。
- submission `displayScore`：12 点单点分平均，即 raw。
- `timeUsed` ≈ `sum(tk_time_ms) * 1000`，判断真实性能看它，不要只看 raw。

解析脚本：

```python
import sys, json, base64, re
sys.path.insert(0, "/home/sakimi26/xpuoj-p1/scripts")
from xpuoj_api import Client

c = Client()
d = c.get_detail(117300).json()
for h, v in d["progress"]["testcaseResult"].items():
    m = re.search(r"OJRESULT v1 \S+ (\S+)", v.get("userOutput", ""))
    if m:
        j = json.loads(base64.b64decode(m.group(1)))
        print(v["input"].strip(), v["status"], v.get("displayScore"),
              j.get("tk_time_ms"), j.get("tb_time_ms"))
```

## 7. userError 诊断优先级

失败时先看每个 case 的 `userError`，按关键词 grep：

```text
Execution error          -> Python 异常，查看 traceback 末尾
TypeError / NameError / UnboundLocalError
CUDA error
SQNR                     -> 数值精度不足；查看实际 dB
DETERMINISM FAIL         -> 两次运行输出不一致
total problem time limit -> 总 500 秒超时（可能是编译慢，也可能是 kernel 太慢）
Pointer argument         -> Triton 指针/launch 参数错误
PassManager              -> Triton 编译 pass 错误
TensorGuardError         -> 沙箱禁止的 API
OutOfResources           -> shared memory / registers 超限
```

通用诊断命令（case 序号按题目改成目标值）：

```python
# 打印目标 case 的 userError 尾部
for k, v in d["progress"]["testcaseResult"].items():
    if v["input"].strip() == "<目标case序号>":
        print(v["userError"][-3000:])
```

## 8. curl 等价示例（凭据自行填充）

```bash
TOKEN=$(curl -s -X POST "$API/auth/login" -H 'Content-Type: application/json' \
  -d @/home/sakimi26/xpuoj-p1/.secrets/xpuoj.json | jq -r .token)

curl -s -X POST "$API/contest/play/submit" \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"contestId":13,"problemOrder":1,"content":{"language":"<题目要求的 language>","code":"...","compileAndRunOptions":{}}}'
```

实际工作直接用 Python 脚本即可。
