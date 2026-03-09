# homework 3

## k-means and knn

- running k-means and knn on `dailySteps.csv`:

```
 k-means:
  custom centers: [2757, 7012, 11605]
  sklearn centers: [678, 5513, 11163]

 k-nearest neighbors:
  testing both models with a new data point: 5000 steps
  custom prediction (k=5): cluster 2
  sklearn prediction (k=5): cluster 2
```

here, my handwritten `kmeans()` and `knn()` functions are compared against the ones in `scikit-learn`, and they return similar values. there is some variance in the k-means center assignments, but that is expected as the initial assignments are random.

## change point analysis

- finding change points in `dailySteps.csv`:

```
identified 8 change points:
  change point 1: index   1 | date: 12/15/2012 | steps shifted at: 6446
  change point 2: index   2 | date: 12/16/2012 | steps shifted at: 4352
  change point 3: index   3 | date: 12/17/2012 | steps shifted at: 10665
  change point 4: index   4 | date: 12/18/2012 | steps shifted at: 5093
  change point 5: index 230 | date: 8/17/2013 | steps shifted at: 14027
  change point 6: index 231 | date: 8/18/2013 | steps shifted at: 311
  change point 7: index 232 | date: 8/20/2013 | steps shifted at: 3220
  change point 8: index 498 | date: 10/11/2014 | steps shifted at: 2745
```
