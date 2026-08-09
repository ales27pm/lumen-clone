# Runtime Evidence Privacy Quarantine

All historical runtime/evidence exports were removed from the current tree on 2026-08-09 because their legacy or unversioned formats could persist raw free-form prompts, responses, events, tool-derived content, or stable diagnostic identifiers. Five exports were confirmed to contain contact, calendar, reminder, or photo-derived personal data. Parse diagnostics, behavior audits, and persistent-runtime exports were also quarantined because their formats cannot enforce the repository's privacy boundary. Their outcome must not be reconstructed by editing failures into successes; any replacement evidence must come from a fresh run using a versioned privacy-safe exporter.

| Removed path | SHA-256 of quarantined bytes |
| --- | --- |
| `2026-06-08-first-improve-loop/1-lumen-live-e2e-report-2026-06-08T23-26-05Z-d03db838-5fe7-40d4-9753-e50426107846.json` | `2be5c7ec25ce30df1a69644ea83ba19ff474a88ef9e3950b5a2babd05f7b0bc1` |
| `2026-06-08-first-improve-loop/5-lumen-live-e2e-report-2026-06-08T23-26-36Z-7c534319-bd66-4cec-9eff-59f850102226.json` | `efaccbc75f444712b646135ce9ebbb5f55e2d8902a668ff2e70559617059c7fb` |
| `2026-06-08-second-improve-loop/2-lumen-live-e2e-report-2026-06-08T23-50-09Z-3eef925b-2fbc-4041-833b-4acaefb80ecf.json` | `3966f1be509acb5b5c8768d222300cb049f821839514b592ec85aace60173ed2` |
| `2026-06-08-second-improve-loop/3-lumen-live-e2e-report-2026-06-08T23-49-46Z-dae44325-1c08-40c5-9814-48166a13612a.json` | `dde64a881ced52e33add237bf73c4f7b21feb1d22ba1833f05c0c2e85b0e9b92` |
| `2026-06-09-third-improve-loop/1-lumen-live-e2e-report-2026-06-09T01-55-20Z-1ac80310-1c2b-4186-b3bb-16b7fc0477aa.json` | `4173a0cb862a17a81b725b4a847f8b98592e3711aae32090d373205ca49a8437` |
| `2026-06-09-third-improve-loop/2-lumen-live-e2e-report-2026-06-09T01-54-50Z-060d6c5a-cd93-4607-a994-509623cd5221.json` | `6ede6a20103ceb39382f209976db09a48d1c6cbf1b4665186b328ded3379c3ee` |
| `2026-06-25-device-e2e/1-latest-e2e-report.json` | `1ac3e01eb8f3c60091316da9745e89f517bc16c1878798deb866898cfc2e45f2` |
| `2026-06-25-device-e2e/2-e2e-results.jsonl` | `2406a809e783e19acb78df060fb00997d8ac72e256e59245cf17507a4b714f43` |
| `2026-06-25-device-e2e/3-latest-e2e-report.txt` | `0b34282fd4f53f35d3356f72e6968d498baf57ffbd19e01bf1da64467a53fa76` |
| `2026-06-25-device-e2e/6-lumen-live-e2e-report-2026-06-24T12-15-25Z-62153bc1-eca5-4f89-bda7-8c5f091cdba9.json` | `5481bd876298264fde0baaca05b865fdc750051b69348f149e41d1a15017bb2c` |
| `2026-06-25-device-e2e/7-agent-behavior-traces.jsonl` | `4dc73a98ed2cc993849fc35b7b06687836eeb8795595a7a10177ebea2fac8ac5` |
| `2026-07-02-latest-e2e/agent-behavior-traces.jsonl` | `a8373ab375b13025b2146a314cb7ff3adf43f45e5ce546e8eee6765281054e17` |
| `2026-07-02-latest-e2e/e2e-results.jsonl` | `4d17ce9ac9be95bfbca6eb590da7b04a4649139a4b28d610602b53a80db59826` |
| `2026-07-02-latest-e2e/lumen-live-e2e-report-2026-07-02T02-09-09Z-451c37fb-3f98-4b39-a8b1-47ffd5082e0c.json` | `c52d7a025bcf64984c672a522a293dc6c7cadc97a929eb6235af6c022dc7274e` |
| `2026-07-02-latest-e2e/lumen-live-e2e-report-2026-07-02T02-19-14Z-e470bc49-d9c1-47c4-ab51-1920b1531b9d.json` | `38b6b7200574f5c962b9415d0da0f253a2c48c7989301e954d86d58609aec7f0` |
| `latest-e2e-report.txt` | `16e72cc3a32df8bc283135ae8ecdaa383d0203c42cbcbbeb0ad3c96cd4e7a125` |
| `2026-06-08-first-improve-loop/2-persistent-runtime-diagnostics-export-10.json` | `f3c87666c1aee75d53b15927e9b18b0efe22a44471c1492db3596f35b6b2a5bb` |
| `2026-06-08-first-improve-loop/3-persistent-runtime-diagnostics-export-9.json` | `7fa30247a909fcdec7dad63f1fbf413b4edff0b821b53bb83fa605673070c33f` |
| `2026-06-08-first-improve-loop/4-lumen-agent-grounding-audit-2026-06-08T23-27-42Z-45583a97-b088-4873-a672-a473ffc46e72.json` | `505fb97d75ba533f6ecb0d5676845cefac79969a33ca089728c408d907d5bea2` |
| `2026-06-08-second-improve-loop/1-persistent-runtime-diagnostics-export.json` | `71d66726b15239044c2be9590a90da8d2541d99d1ecb5b259512b6eaf19c48de` |
| `2026-06-08-second-improve-loop/4-lumen-agent-grounding-audit-2026-06-08T23-56-02Z-db8e545a-8842-497c-a87d-de92f0d8a827.json` | `14b91935ae7e031f734bfba6040576c346f738b7ab2e33573cf3bc7651db63d3` |
| `2026-06-08-second-improve-loop/5-lumen-agent-grounding-audit-2026-06-08T23-49-25Z-a5deb496-392f-4882-bc15-539f86168527.json` | `61e30bc79bd0b17002eda59c0b33dde5650c2157616e9278404e5b0bb904038f` |
| `2026-06-09-third-improve-loop/3-lumen-agent-grounding-audit-2026-06-09T01-55-41Z-bac56b76-988d-4475-9f8f-0c451b0bae5d.json` | `6091e7c2aea714d7fa5c33a3bd78dd6a2bc3fa97fd34ad22a049ccf751eae3c0` |
| `2026-06-09-third-improve-loop/4-persistent-runtime-diagnostics-export-11.json` | `a8cb0aedd58c3d9047f0e47640587ecf26d73e03b6a31cf52eff320e311399f8` |
| `2026-06-09-third-improve-loop/5-lumen-agent-grounding-audit-2026-06-09T01-51-37Z-70e09f27-5596-4087-a606-06242a27456d.json` | `aead4de51078a64eaadd502751a47b84165932e21e9644998691bb9c781c7d4b` |
| `2026-06-25-device-e2e/4-agent-parse-noise.jsonl` | `a7e9eb289a6aa7a72639a654cd06e878f954dc0f7919f2b9b93ab37e05fb8452` |
| `2026-06-25-device-e2e/5-agent-parse-failures.jsonl` | `f11fefb34139df6cbec34a99ec338070b6372bb98197515bc490366b7e0f2ac7` |
| `2026-07-02-latest-e2e/agent-parse-failures.jsonl` | `505a632bcee7de43ddc2d4ecf1da0cc5d97e132d911d5a8907d382b3d371a12d` |
| `2026-07-02-latest-e2e/agent-parse-noise.jsonl` | `f775bc9850e1c80e3b7431ef499a4ea030b519105c616f49f8eaf0b0f94ffe64` |

This removal contains the current branch, not prior Git objects or remote clones. A history rewrite and coordinated remote cleanup require a separate explicit decision because they invalidate existing commit identities and force downstream clones to resynchronize.
