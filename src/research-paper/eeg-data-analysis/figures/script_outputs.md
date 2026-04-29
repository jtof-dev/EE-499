# `train_singlem.py`

```
❯ uv run train_singlem.py 
loading silent EEG data for train...
loaded 3005 clean train windows from 13 files
loading silent EEG data for val...
loaded 1131 clean val windows from 5 files

training on device: cuda
info: partial fine-tuning enabled. 36 backbone tensors will train with a micro-learning rate.
beginning training loop...
epoch [01/50] | train loss: 0.6613, acc: 62.4% | val loss: 0.6840, acc: 61.5%
epoch [02/50] | train loss: 0.6155, acc: 69.4% | val loss: 0.6406, acc: 64.1% epoch [03/50] | train loss: 0.5776, acc: 71.9% | val loss: 0.6210, acc: 66.0%
epoch [04/50] | train loss: 0.5146, acc: 77.9% | val loss: 0.5824, acc: 73.0%
epoch [05/50] | train loss: 0.4517, acc: 83.5% | val loss: 0.5823, acc: 73.4%
epoch [06/50] | train loss: 0.4199, acc: 86.2% | val loss: 0.5808, acc: 73.7%
epoch [07/50] | train loss: 0.3992, acc: 87.7% | val loss: 0.6299, acc: 72.1%
no improvement in val loss. early stopping counter: 1/15
epoch [08/50] | train loss: 0.3805, acc: 88.8% | val loss: 0.6136, acc: 71.4%
no improvement in val loss. early stopping counter: 2/15
epoch [09/50] | train loss: 0.3700, acc: 90.0% | val loss: 0.6263, acc: 71.6%
no improvement in val loss. early stopping counter: 3/15
epoch [10/50] | train loss: 0.3559, acc: 90.7% | val loss: 0.5946, acc: 74.2%
no improvement in val loss. early stopping counter: 4/15
epoch [11/50] | train loss: 0.3500, acc: 91.2% | val loss: 0.5932, acc: 73.5%
no improvement in val loss. early stopping counter: 5/15
epoch [12/50] | train loss: 0.3354, acc: 92.5% | val loss: 0.6060, acc: 73.7%
no improvement in val loss. early stopping counter: 6/15
epoch [13/50] | train loss: 0.3322, acc: 92.6% | val loss: 0.6148, acc: 72.2%
no improvement in val loss. early stopping counter: 7/15
epoch [14/50] | train loss: 0.3215, acc: 93.6% | val loss: 0.6240, acc: 72.8%
no improvement in val loss. early stopping counter: 8/15
epoch [15/50] | train loss: 0.3196, acc: 93.5% | val loss: 0.6244, acc: 73.2%
no improvement in val loss. early stopping counter: 9/15
epoch [16/50] | train loss: 0.3088, acc: 94.6% | val loss: 0.6372, acc: 71.4%
no improvement in val loss. early stopping counter: 10/15
epoch [17/50] | train loss: 0.3056, acc: 94.9% | val loss: 0.6702, acc: 71.4%
no improvement in val loss. early stopping counter: 11/15
epoch [18/50] | train loss: 0.3006, acc: 95.0% | val loss: 0.6594, acc: 72.1%
no improvement in val loss. early stopping counter: 12/15
epoch [19/50] | train loss: 0.2980, acc: 95.2% | val loss: 0.6367, acc: 72.9%
no improvement in val loss. early stopping counter: 13/15
epoch [20/50] | train loss: 0.2936, acc: 96.0% | val loss: 0.6641, acc: 71.6%
no improvement in val loss. early stopping counter: 14/15
epoch [21/50] | train loss: 0.2905, acc: 95.4% | val loss: 0.6696, acc: 72.3%
no improvement in val loss. early stopping counter: 15/15

early stopping triggered at epoch 21.

training complete
optimal weights found at epoch 6
best validation loss: 0.5808
corresponding validation accuracy: 73.7%
model saved to 'singlem_binary_head.pth'
```

# `singlem_significance.py`

```
extracting neural features for andy (stroop)...

EEG predicted anxiety: statistical analysis

shapiro-wilk test (normality)
  silent      : normal (p = 0.8749, n = 6)
  whitenoise  : normal (p = 0.4742, n = 3)
  music       : normal (p = 0.1252, n = 3)
  musicnl     : normal (p = 0.5338, n = 3)

levene's test (equal variance)
  result      : equal variances (p = 0.2624)

main significance test
  method: one-way ANOVA (data passed all pre-flight checks)
  no significant difference (p = 0.9643)
  the model's predicted anxiety levels did not differ significantly between audio conditions.
```

```
extracting neural features for andy (typing)...

EEG predicted anxiety: statistical analysis

shapiro-wilk test (normality)
  silent      : normal (p = 0.8559, n = 6)
  whitenoise  : normal (p = 0.2963, n = 3)
  music       : normal (p = 0.9229, n = 3)
  musicnl     : normal (p = 0.4875, n = 3)

levene's test (equal variance)
  result      : equal variances (p = 0.8134)

main significance test
  method: one-way ANOVA (data passed all pre-flight checks)
  no significant difference (p = 0.2914)
  the model's predicted anxiety levels did not differ significantly between audio conditions.
```

# `metrics_stroop_graphs.py`

```
processing metrics for participant matching: ^andy$

condition    | runs  | avg keys   | avg correct  | avg errors | keys/sec   | accuracy %
silent       | 6     | 458.5      | 431.7        | 26.8       | 1.54       | 94.15     
whitenoise   | 3     | 453.7      | 428.7        | 25.0       | 1.52       | 94.49     
music        | 3     | 480.3      | 442.0        | 38.3       | 1.61       | 92.02     
musicnl      | 3     | 531.7      | 500.7        | 31.0       | 1.78       | 94.17  
```

# `metrics_typing_graphs.py`

```
processing typing metrics for participant matching: ^andy$

condition    | runs  | avg words  | avg errors | words/sec  | accuracy %
silent       | 6     | 241.5      | 54.5       | 0.81       | 81.59     
whitenoise   | 3     | 266.3      | 49.7       | 0.89       | 84.28     
music        | 3     | 267.7      | 57.7       | 0.90       | 82.27     
musicnl      | 3     | 268.7      | 50.0       | 0.90       | 84.31 
```

# `metrics_significance.py`

```
loading data and running final statistical analysis for andy on stroop test...


shapiro-wilk test (are the distributions normal?)
null hypothesis: the data is normally distributed (p > 0.05)


levene's test (is the variance roughly equal across groups?)
null hypothesis: all groups have equal variance (p > 0.05)


next steps:
if all conditions are normal and have equal variances -> run an ANOVA.
if any condition is not normal or variance is unequal -> run a kruskal-wallis test.

--------------------------------------------------

>>> analyzing accuracy <<<
shapiro (silent): normal (p = 0.5202)
shapiro (whitenoise): normal (p = 0.3650)
shapiro (music): normal (p = 0.4187)
shapiro (musicnl): normal (p = 0.5450)
levene (all groups): equal variances (p = 0.7196)

accuracy (method: one-way ANOVA)
no significant difference (p = 0.6535)
any variations between audio conditions are likely just random chance.
--------------------------------------------------

>>> analyzing throughput <<<
shapiro (silent): normal (p = 0.2111)
shapiro (whitenoise): normal (p = 0.1981)
shapiro (music): normal (p = 0.4294)
shapiro (musicnl): normal (p = 0.5857)
levene (all groups): unequal variances (p = 0.0131)

throughput (method: kruskal-wallis h test)
no significant difference (p = 0.1149)
any variations between audio conditions are likely just random chance.
--------------------------------------------------
```

```
loading data and running final statistical analysis for andy on typing test...


shapiro-wilk test (are the distributions normal?)
null hypothesis: the data is normally distributed (p > 0.05)


levene's test (is the variance roughly equal across groups?)
null hypothesis: all groups have equal variance (p > 0.05)


next steps:
if all conditions are normal and have equal variances -> run an ANOVA.
if any condition is not normal or variance is unequal -> run a kruskal-wallis test.

--------------------------------------------------

>>> analyzing accuracy <<<
shapiro (silent): normal (p = 0.6001)
shapiro (whitenoise): normal (p = 0.4888)
shapiro (music): normal (p = 0.9775)
shapiro (musicnl): normal (p = 0.6597)
levene (all groups): equal variances (p = 0.1960)

accuracy (method: one-way ANOVA)
no significant difference (p = 0.0781)
any variations between audio conditions are likely just random chance.
--------------------------------------------------

>>> analyzing throughput <<<
shapiro (silent): normal (p = 0.0856)
shapiro (whitenoise): normal (p = 0.1878)
shapiro (music): normal (p = 0.8632)
shapiro (musicnl): normal (p = 0.1982)
levene (all groups): unequal variances (p = 0.0296)

throughput (method: kruskal-wallis h test)
significant difference found (p = 0.0379)
audio condition had a measurable impact.
(note: for strict non-parametric pairwise comparisons, dunn's test via the scikit-posthocs library is recommended).
--------------------------------------------------
```

# `EEG_band_analysis_and_significance.py`

```
Scanning 'data/level_2' for participant 'andy'...

Aggregated Predicted PSS Levels
test_task  condition  runs  avg_predicted_pss
   Stroop      Music     3             -0.935
   Stroop    MusicNL     3             -1.331
   Stroop     Silent     6             -0.863
   Stroop WhiteNoise     3             -1.293
   Typing      Music     3             -1.799
   Typing    MusicNL     3             -1.359
   Typing     Silent     6             -0.376
   Typing WhiteNoise     3             -1.025

=============================================
STATISTICAL ANALYSIS
=============================================

--- Analysis for Task: STROOP ---
Normality check (Shapiro-Wilk):
   - music     : p = 0.8081 [pass]
   - whitenoise: p = 0.9032 [pass]
   - silent    : p = 0.6256 [pass]
   - musicnl   : p = 0.1194 [pass]

Variance check (Levene's):
   - all groups: p = 0.6433 [pass]

Assumptions met. Running parametric One-Way ANOVA:

anova Results:
Statistic: 0.926
p-value:   0.4603
Conclusion: No significant difference detected for the STROOP task.

--- Analysis for Task: TYPING ---
Normality check (Shapiro-Wilk):
   - musicnl   : p = 0.7999 [pass]
   - music     : p = 0.1073 [pass]
   - silent    : p = 0.6970 [pass]
   - whitenoise: p = 0.2278 [pass]

Variance check (Levene's):
   - all groups: p = 0.3843 [pass]

Assumptions met. Running parametric One-Way ANOVA:

anova Results:
Statistic: 2.058
p-value:   0.1642
Conclusion: No significant difference detected for the TYPING task.```
