# S1 v2 SID153526 complete-AC anomaly audit

Exact frozen sourceSHA `d11bf076b18ff243984aa346b551c3c1bcbe8c0ba65a9bc2c18ad927c8ad3497` verified against153526-source.py and candidate. Original rawmetaAccepted/display87.75;12casesAccepted/pass=true, twoSQNR>=22 and two determinism passes each, minimum22.69dB. Raw87.75/net77.75 is **not a new high**; best153151 remainsraw89.75/net79.75. No production promotion.

Exactzeros c11,c12;lowthreshold cases c9–c12. qsum1053,Σtk22.721ms,rawtimeUsed22721 is consistent withΣtk*1000, not wholewalltime.

| c | tb ms | tk ms | platformq | minSQNR dB |
|---|---:|---:|---:|---:|
| 1 | 17.686 | 4.603 | 79 | 23.12 |
| 2 | 28.721 | 7.907 | 78 | 23.14 |
| 3 | 6.951 | 1.344 | 83 | 22.70 |
| 4 | 4.942 | 0.775 | 86 | 22.98 |
| 5 | 13.791 | 2.739 | 83 | 23.13 |
| 6 | 7.480 | 1.200 | 86 | 22.93 |
| 7 | 10.259 | 2.096 | 83 | 23.12 |
| 8 | 7.209 | 1.142 | 86 | 22.90 |
| 9 | 7.743 | 0.852 | 90 | 23.13 |
| 10 | 7.733 | 0.063 | 99 | 22.69 |
| 11 | 5.656 | 0.000 | 100 | 23.14 |
| 12 | 8.152 | 0.000 | 100 | 22.90 |

Targetc6tk1.200 vs normal1535071.249 is−3.92%; however untouchedc7tk2.096vs2.183 is−3.99%, anduntouchedc8tk1.142vs1.231 is−7.23%. Those shared changes prevent claiming stablec6structural acceleration from this one run. c5tb13.791vs12.085 andc10tb7.733vs6.693 are also elevated. Laterlow/zero samples are separate anomaly observations, not normalthroughput evidence. No paired normalc6 measurement establishesgain here; production staysf9ca.

This is a new complete-AC exact-zero sample on a different frozen source, not evidence of a deterministic zero trigger or repetition probability. Preserve original/audit/source and collect future same-SHA/normalcontrols only asrootdirects. No timing probe, new launch or additional historical scan was performed.
