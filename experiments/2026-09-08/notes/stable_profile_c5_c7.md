# stable c5/c7 replay probes

基底为 `stable_shape_dispatch.py`，SHA-256 `67340914ae3759c84dd541ec5cece0f8305fdeb8a517286bc8105fc05128f9ea`。

- `stable_profile_md_c5_c7.py`：SHA-256 `bf7066fac0bc6332b9e754a9cd48d457a6d88ce633ab21b35418fb374b9e845e`
- `stable_profile_dn_c5_c7.py`：SHA-256 `cbc39618f786ba1e432bfaada768476c47258bb203956b031ddda28c9f747011`

两份探针都只触达 c5 `(8192,3584,64,2560,8)` 与 c7 `(16384,4096,96,2048,3)`。SID 141397 首调 TLE 后，这两份全量 stable 底盘探针已冻结且未提交。

partial 底盘 `stable_c5_c7_dispatch.py` 的 SHA-256 为 `64ad7cf0f5b9a1c39747c326310d0084e259bd96681b4cc546eede85f73c1b50`。SID 141408 全 12 案 Accepted，可作诊断底盘但未晋升：

- `c57_profile_md.py`：SHA-256 `2d2435694e8fef0801f1ceb7766057450f05738b5c7f236acc65815fc5fdcdd7`
- `c57_profile_dn.py`：SHA-256 `cf63d3bd4022245315840fc3b6b5d4a3962a36837d748e962ededf270c1a57d7`

两份 partial 探针同样只触达 c5/c7。MD 的 SID 141410 全 12 案 Accepted，新增增量为 c5 约 1.748–1.750 ms、c7 约 1.310–1.312 ms；DN 的 SID 141411 全 12 案 Accepted，新增增量为 c5 约 0.861–0.864 ms、c7 约 0.651–0.653 ms。两者仅用于固定路径分段测量。

`md_bn64_two_cta_c57.py` 的 SID 141413 在两次 SQNR 后 TLE、没有目标数据；同码 SID 141415 全 12 案 Accepted，但校正后 c5 慢约 22.5–23.3%、c7 慢约 13.9–14.6%，该路线已淘汰且不再扫参数。题面只保证同一测试点内权重静态，切换测试点后可能变化；现有 shape-only 权重缓存是晋升前必须修正的缺口。
