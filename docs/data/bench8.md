| Corpus | Query | Lab 3 Python scorer, ms | NumPy, ms | Speed-up |
|---|---|---|---|---|
| real, 91 docs | `the` | 9.660 | 0.048 | 203.2x |
| real, 91 docs | `mersenne` | 0.016 | 0.025 | 0.6x |
| real, 91 docs | `whale OR harpoon OR ship` | 0.213 | 0.080 | 2.7x |
| synthetic, 100 000 docs | `common` | 90.036 | 2.942 | 30.6x |
| synthetic, 100 000 docs | `rare` | 0.065 | 0.066 | 1.0x |
| synthetic, 100 000 docs | `common OR mid OR rare` | 98.046 | 3.883 | 25.3x |
