
semantic query: 2.8 ms (embed query + 10,275 x 384 matrix product + per-doc max)
| # | Query | Relevant | BM25 top 5 | P@5 | Semantic top 5 | P@5 | Hybrid top 5 | P@5 |
|---|---|---|---|---|---|---|---|---|
| 1 | `frankenstein OR monster` | 74, 87 | 74, 87, 6, 3, 27 | 0.4 | 74, 27, 54, 87, 68 | 0.4 | 74, 87, 27, 54, 3 | 0.4 |
| 2 | `paradise OR satan OR eden` | 11, 17, 48 | 11, 17, 48, 8, 24 | 0.6 | 48, 17, 24, 11, 13 | 0.6 | 48, 17, 11, 24, 8 | 0.6 |
| 3 | `internet OR email OR network` | 30, 39, 46, 65, 70 | 70, 25, 29, 39, 46 | 0.6 | 65, 43, 25, 70, 30 | 0.6 | 70, 25, 65, 43, 39 | 0.6 |
| 4 | `census OR population` | 5, 20, 28, 38 | 28, 20, 9, 57, 38 | 0.6 | 28, 20, 16, 77, 55 | 0.4 | 28, 20, 16, 77, 57 | 0.4 |
| 5 | `prime OR primes OR mersenne` | 55, 59 | 59, 72, 55, 38, 5 | 0.4 | 59, 55, 29, 20, 68 | 0.4 | 59, 55, 29, 56, 81 | 0.4 |
| 6 | `lord OR god OR commandments` | 0, 8, 21 | 8, 0, 21, 48, 88 | 0.6 | 21, 0, 8, 48, 22 | 0.6 | 8, 21, 0, 48, 22 | 0.6 |
| 7 | `guillotine OR paris OR revolution` | 50, 89 | 50, 89, 60, 24, 71 | 0.4 | 89, 50, 51, 16, 1 | 0.4 | 50, 89, 13, 60, 76 | 0.4 |
| 8 | `slavery OR freedom OR slaves` | 14, 57, 90 | 57, 14, 61, 90, 76 | 0.6 | 14, 57, 76, 90, 61 | 0.6 | 14, 57, 61, 76, 90 | 0.6 |
| 9 | `anne OR gilbert OR marilla` | 35, 42 | 35, 42, 37, 16, 38 | 0.4 | 35, 37, 42, 50, 66 | 0.4 | 35, 37, 42, 72, 34 | 0.4 |
| 10 | `constitution OR union OR states` | 9, 80 | 9, 38, 77, 16, 5 | 0.2 | 80, 61, 9, 40, 90 | 0.4 | 9, 80, 38, 16, 40 | 0.4 |
| 11 | `giant sea mammal hunted by men with spears on wooden boats` *(paraphrase)* | 6 |  | 0.0 | 6, 12, 81, 75, 83 | 0.2 | 6, 81, 75, 71, 83 | 0.2 |
| 12 | `the fallen angel rebels against heaven` *(paraphrase)* | 11, 17, 48 | 1, 17, 60, 0, 21 | 0.2 | 11, 17, 48, 21, 8 | 0.6 | 17, 11, 21, 8, 0 | 0.4 |
| 13 | `how to send electronic letters across computer networks` *(paraphrase)* | 30, 39, 46, 65, 70 | 70, 43, 25, 65, 29 | 0.4 | 65, 70, 25, 30, 39 | 0.8 | 70, 65, 25, 43, 39 | 0.6 |
| 14 | `official count of every resident of the nation` *(paraphrase)* | 20, 28 | 16, 38, 5, 77, 70 | 0.0 | 5, 40, 16, 28, 20 | 0.4 | 5, 16, 38, 77, 70 | 0.0 |
| 15 | `a young orphan girl growing up on a farm in canada` *(paraphrase)* | 35, 42 | 35, 37, 29, 60 | 0.2 | 52, 67, 14, 15, 24 | 0.0 | 35, 37, 15, 14, 67 | 0.2 |
| | **Mean, 10 lab-3 queries** | | | **0.48** | | **0.48** | | **0.48** |
| | **Mean, 5 paraphrases** | | | **0.16** | | **0.40** | | **0.28** |
| | **Mean, all 15** | | | **0.37** | | **0.45** | | **0.41** |
