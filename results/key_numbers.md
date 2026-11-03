t_compute per step (real 4-process run): 141.4 ms

## Estimated minutes for 2,000 steps + final val loss (mean of seeds 1-2)

| label                       |   regional |   intercontinental |   constrained |   poor |   final_val |
|:----------------------------|-----------:|-------------------:|--------------:|-------:|------------:|
| DDP (sync every step)       |      21.67 |              48.78 |        186.23 | 907.14 |        1.59 |
| Hierarchical H=10/50, INT8  |       4.79 |               5.12 |          5.44 |   7.1  |        1.66 |
| Hierarchical H=10/50        |       4.91 |               5.41 |          6.59 |  13.29 |        1.67 |
| Hierarchical H=10/50, INT4  |       4.78 |               5.03 |          5.38 |   6.77 |        1.68 |
| DiLoCo H=10                 |       6.41 |               9.37 |         22.91 |  93.06 |        1.69 |
| DiLoCo H=50                 |       5.05 |               5.65 |          8.35 |  22.38 |        1.7  |
| DiLoCo H=50, INT4           |       4.77 |               5    |          5.51 |   7.48 |        1.7  |
| DiLoCo H=50, INT8           |       4.84 |               5.17 |          5.94 |   9.29 |        1.7  |
| Hierarchical H=10/100       |       4.83 |               5.07 |          5.67 |   9.02 |        1.78 |
| Hierarchical H=10/100, INT8 |       4.77 |               4.93 |          5.09 |   5.92 |        1.79 |
| Hierarchical H=10/100, INT4 |       4.76 |               4.88 |          5.06 |   5.75 |        1.79 |
| DiLoCo H=100                |       4.88 |               5.18 |          6.53 |  13.55 |        1.83 |
| DiLoCo H=100, INT4          |       4.74 |               4.86 |          5.11 |   6.1  |        1.84 |
| DiLoCo H=100, INT8          |       4.78 |               4.94 |          5.33 |   7    |        1.86 |

## Median sync round (s)

|                                  |       4 |       8 |     32 |
|:---------------------------------|--------:|--------:|-------:|
| ('constrained', 'flat')          |   1.186 |   1.831 |  5.458 |
| ('constrained', 'flat_b2b')      | nan     | nan     |  5.446 |
| ('constrained', 'hier')          |   0.975 |   1.06  |  2.79  |
| ('intercontinental', 'flat')     |   0.424 |   0.687 |  1.398 |
| ('intercontinental', 'flat_b2b') | nan     | nan     |  1.322 |
| ('intercontinental', 'hier')     |   0.439 |   0.574 |  1.012 |
| ('local', 'lan')                 | nan     | nan     |  0.008 |
| ('poor', 'flat')                 |   4.142 |   6.863 | 26.503 |
| ('poor', 'flat_b2b')             | nan     | nan     | 27.073 |
| ('poor', 'hier')                 |   3.048 |   3.542 | 12.837 |
| ('regional', 'flat')             |   0.082 |   0.194 |  0.508 |
| ('regional', 'flat_b2b')         | nan     | nan     |  0.509 |
| ('regional', 'hier')             |   0.061 |   0.09  |  0.267 |

## Final val loss per config (seeds)

| name                | method   |   H |   H_r |   H_g |   bits |   wan_syncs |   lan_syncs |   steps |   mean |    min |    max |   count |
|:--------------------|:---------|----:|------:|------:|-------:|------------:|------------:|--------:|-------:|-------:|-------:|--------:|
| ddp                 | ddp      |  50 |    10 |    50 |     32 |        2000 |           0 |    2000 | 1.5871 | 1.5815 | 1.5927 |       2 |
| diloco_H100_b32     | diloco   | 100 |    10 |    50 |     32 |          20 |           0 |    2000 | 1.8265 | 1.8076 | 1.8454 |       2 |
| diloco_H100_b4      | diloco   | 100 |    10 |    50 |      4 |          20 |           0 |    2000 | 1.8412 | 1.8219 | 1.8606 |       2 |
| diloco_H100_b8      | diloco   | 100 |    10 |    50 |      8 |          20 |           0 |    2000 | 1.8646 | 1.839  | 1.8901 |       2 |
| diloco_H10_b32      | diloco   |  10 |    10 |    50 |     32 |         200 |           0 |    2000 | 1.6887 | 1.679  | 1.6984 |       2 |
| diloco_H50_b32      | diloco   |  50 |    10 |    50 |     32 |          40 |           0 |    2000 | 1.6962 | 1.694  | 1.6983 |       2 |
| diloco_H50_b4       | diloco   |  50 |    10 |    50 |      4 |          40 |           0 |    2000 | 1.7027 | 1.6873 | 1.7181 |       2 |
| diloco_H50_b8       | diloco   |  50 |    10 |    50 |      8 |          40 |           0 |    2000 | 1.7035 | 1.6905 | 1.7164 |       2 |
| hier_Hr10_Hg100_b32 | hier     |  50 |    10 |   100 |     32 |          20 |         200 |    2000 | 1.7794 | 1.7703 | 1.7885 |       2 |
| hier_Hr10_Hg100_b4  | hier     |  50 |    10 |   100 |      4 |          20 |         200 |    2000 | 1.7913 | 1.7908 | 1.7917 |       2 |
| hier_Hr10_Hg100_b8  | hier     |  50 |    10 |   100 |      8 |          20 |         200 |    2000 | 1.7862 | 1.7857 | 1.7867 |       2 |
| hier_Hr10_Hg50_b32  | hier     |  50 |    10 |    50 |     32 |          40 |         200 |    2000 | 1.6653 | 1.6608 | 1.6698 |       2 |
| hier_Hr10_Hg50_b4   | hier     |  50 |    10 |    50 |      4 |          40 |         200 |    2000 | 1.6824 | 1.6808 | 1.684  |       2 |
| hier_Hr10_Hg50_b8   | hier     |  50 |    10 |    50 |      8 |          40 |         200 |    2000 | 1.6611 | 1.6558 | 1.6663 |       2 |

## End-to-end distributed runs

- constrained 8-bit: final val 2.5168; rounds [3.998, 2.307, 2.278, 2.505, 2.916, 2.498]; t_compute/step 138.5 ms; total 61.5 s
- constrained 32-bit: final val 2.5339; rounds [6.719, 5.41, 5.79, 5.791, 6.361, 6.199]; t_compute/step 144.4 ms; total 82.7 s
