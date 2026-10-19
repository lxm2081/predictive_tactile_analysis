# Algorithm specification

## Dataset and split

The measured dataset contains 30 trials, each with 500 frames from a 44 x 44 tactile array
sampled at 200 Hz. Six normal trials are used for fitting. Two normal trials and one trial
from each anomaly type form the validation split. The held-out test split contains two
normal trials and three trials from each of four anomaly types. Four end-stop trials are
treated as task-completion controls and are excluded from anomaly-positive ROC evaluation.

The first 80 frames form a fixed initialization interval. Model fitting, residual-scale
estimation, threshold selection and held-out scoring use transitions beginning at score
index 80. Score index j compares the prediction made from frame j with measured frame j+1.
At 200 Hz and 5 mm/s, the excluded interval corresponds to 0.40 s and 2.0 mm of travel.
The predictor is fitted offline and remains fixed; the initialization interval is not an
online learning period.

## Eight-dimensional tactile code

For tactile frame P with taxel coordinates x and y, the code contains:

1. total contact intensity, the sum of all taxel values;
2. peak response, the maximum taxel value;
3. effective contact area, the number of taxels above 0.20 times the frame peak;
4. pressure centroid x, weighted by the full nonnegative tactile frame;
5. pressure centroid y, weighted by the full nonnegative tactile frame;
6. spatial dispersion, the pressure-weighted second central moment in x and y;
7. left-right asymmetry, defined as right intensity minus left intensity divided by total intensity;
8. temporal variation, the mean absolute taxel-wise difference from the immediately preceding frame.

The first temporal-variation value in each trial is zero. Coordinates use taxel indices.
Because centroid and dispersion variables are standardized from training data, a fixed affine
coordinate conversion does not alter the fitted or evaluated result.

## Training-only standardization

For feature k, the normal-training mean and population standard deviation are calculated
from frames 80 through 499. Every training, validation, test and control trial is transformed
using these same values.

## One-step predictor

The row-vector convention is

```text
z_hat[t+1] = z[t] W + b
```

W and b are fitted from normal-training transitions t = 80 through 498 using ridge
regression. The bias is not penalized and the ridge coefficient is 0.001.

## Residual-normalized mismatch

For each code coordinate k, the residual standard deviation sigma_k is estimated from the
same normal-training transitions. The score is

```text
r[t+1,k] = (z[t+1,k] - z_hat[t+1,k]) / sigma_k
m[t+1]   = sqrt(mean_k(r[t+1,k]^2))
```

This score is therefore the root-mean-square channel-wise residual z-score.

## Comparison scores

- Unweighted predictive is the RMS prediction residual in the standardized 8-D code.
- Raw frame difference is the RMS taxel-wise change between adjacent 44 x 44 frames.
- Raw-difference deviation is the absolute z-score of raw frame-difference magnitude relative to normal training.
- Static tactile-code template is the RMS distance from the normal-training code mean, which is zero after standardization.
- Tactile-code difference is the absolute z-score of the RMS 8-D code-change magnitude relative to normal training.
- Peak pressure is the maximum value in the measured next frame.

## Frame-level labels and metrics

The event mask stored in the NPZ is authoritative for frame-level ROC labels. Its inclusive
onset and exclusive end were fixed from the synchronized experimental intervention record,
without using the subsequently calculated mismatch scores. For a score comparing frames t
and t+1, the label is the event-mask value at frame t+1. Initialization transitions are
excluded. Overall test metrics use every trial with split equal to test. This gives 654
positive and 5,212 negative transitions across the 14 held-out normal and disturbance trials.
Condition-wise AUC combines the two normal test trials with the three test trials of the
selected anomaly condition. End-stop controls retain an annotated timing window but have
zero-valued ROC masks, so they cannot enter the anomaly-positive ROC calculation.

## Operating point

The mismatch threshold is the 99.5th percentile of monitored mismatch scores from the two
normal validation trials. An alarm requires three consecutive scores strictly above the
threshold. The first-crossing frame is the first frame in the qualifying run. The confirmation
frame is two frames later. Detection distance is calculated from the first-crossing frame,
sampling frequency and trial pulling speed. The annotated onset and exclusive end frame
define the qualifying window for anomaly and end-stop trial-level detection. The detailed
output separately records the first persistent crossing over the complete monitored interval
and flags any crossing completed before the annotated onset. This diagnostic does not alter
the archived window-restricted detection statistic.
