# 21:56轮异常证据索引

| SID / 比较 | SHA | 结果 / 精确零 | 证据 |
|---|---|---|---|
| 历史153151 | `f9ca009609bc…` | 完整AC/raw89.75/net79.75；zero10–12，low8–12；c1tb异常 | [历史审计](../../session-1654/platform/153151-audit.json) |
| 同码153507 | `f9ca009609bc…` | 完整AC/raw81.92/net71.92；zero/low空 | [同SHA对照](SAME_SHA_CONTROL.md)、[原审计](../platform/153507-audit.json) |
| 新153526 S1v2 | `d11bf076b18f…` | 完整AC/raw87.75/net77.75；zero11/12，low9–12；不是新高 | [独立审计](SID153526_AUDIT.md)、[原始](../platform/153526-raw.json)、[audit](../platform/153526-audit.json)、[source](../platform/153526-source.py) |

检查采用完整12案及每案两SQNR/两确定性，通过并不等于正常tk/吞吐或稳定触发。最高分、正常收益与异常证据分开记录；不公开任何key/token字段。
