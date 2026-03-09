# homework 3

## k-means and knn

- running k-means and knn on `dailySteps.csv`:

```
 k-means:
  custom centers: [729, 5798, 12158]
  sklearn centers: [678, 5513, 11163]

 k-nearest neighbors:
  testing both models with new data points (k=5):

  - For point: 2000 steps
    custom prediction:  cluster 0
    sklearn prediction: cluster 1

  - For point: 8000 steps
    custom prediction:  cluster 1
    sklearn prediction: cluster 2

  - For point: 14000 steps
    custom prediction:  cluster 2
    sklearn prediction: cluster 0
```

here, my handwritten `kmeans()` and `knn()` functions are compared against the ones in `scikit-learn`, and they return similar values. there is notable variance between my custom functions and the `scikit-learn` implementations because my `kmeans()` function is a single pass implementation starting from random variables, while the `scikit-learn` function is iterative, and recalculates centroids until it finds a stable solution. then, when running `knn()`, that inaccuracy carries over and causes some variance in the predicted cluster.

## change point analysis

- finding change points in `dailySteps.csv`:

```
identified 8 change points:
change point 1: index 1 | date: 12/15/2012 | steps shifted at: 6446
change point 2: index 2 | date: 12/16/2012 | steps shifted at: 4352
change point 3: index 3 | date: 12/17/2012 | steps shifted at: 10665
change point 4: index 4 | date: 12/18/2012 | steps shifted at: 5093
change point 5: index 230 | date: 8/17/2013 | steps shifted at: 14027
change point 6: index 231 | date: 8/18/2013 | steps shifted at: 311
change point 7: index 232 | date: 8/20/2013 | steps shifted at: 3220
change point 8: index 498 | date: 10/11/2014 | steps shifted at: 2745
```

these change points do make sense relative to their time of year. for change points 1-4, there are multiple large shifts on back-to-back days, likely indicating a mixture of lower activity due to a Christmas break and higher activity from going Christmas shopping. then, points 5-7 show a large spike, dip, then back to about average, likely indicating a summer vacation. finally, point 8 is on the last day of the dataset, possibly because the watch was taken off partway through the day.
