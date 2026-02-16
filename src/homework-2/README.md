# Homework 2

## 1. Daily Steps (FitBit steps data)

- Arithmetic: 10751.6
- Harmonic: 504.8

Using the FitBit data, the four paricipants had an arithmetic mean of 10752 and a harmonic mean of 505. The arithmetic mean is much closer to the actual daily totals steps, while the harmonic mean is essentially a useless statistic here. There are some very low outliers (I remember one day with a total of 4 steps), which affects the harmonic mean much more than the arithmetic mean.

## 2. Group Variance (FitBit steps data)

- Group Pooled Standard Deviation: 21.821
- Group Pooled Variance: 476.151

A pooled standard deviation of 22 means (standard deviation is the square root of the variance) that the four participants walked very similar amounts per day. Considering that the average steps per day was over 10,000, only having a ~20 step difference between participants is suprising.

## 3. Comparing the Devices

- T-Test: 23.159
- P-Value: 0.0

When comparing daily steps data between the FitBit and ActiGraph, the T-Test between the two is 23. This means that there is a significant difference between the two wearables, and a P-Value of 0.0 means that the null hypothesis is not supported (so the difference cannot be due to random chance).

## 4. Weekend Warriors (FitBit steps data)

- F-Stat: 7.897
- P-Value: 5.642e-05

A high F-Stat of 7.8 and very low P-Value means that there is a significant difference in daily steps patterns depending on the day of the week. I added some code that shows this difference:

| Day       | Average Steps |
| --------- | ------------- |
| Monday    | 45277.25      |
| Tuesday   | 48772.20      |
| Wednesday | 51040.60      |
| Thursday  | 44002.20      |
| Friday    | 50436.40      |
| Saturday  | 39539.00      |
| Sunday    | 22432.00      |

(Note: the mean I took here was after all four participants had been combined into one list per day, so that is why these are in the range of 20-50k. However, this spread still makes sense considering that the daily step mean from 1. was 10.7k steps)

Suprisingly, this shows that the participants were NOT weekend warriors, instead walking notably less on weekends instead of more.

## 5. Seasonality

- F-Stat: 0.371
- P-Value: 0.954

A low F-Stat of 0.37 and a high P-Value of almost 1 means that the null hypothesis is true - there is not a significant difference in daily steps patterns depending on the season.