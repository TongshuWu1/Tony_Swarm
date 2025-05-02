# Balloon Detection and Tracking on Nicla Vision

## Repo Structure

```text
ComputerVision/
│
├── get_gain/
│   └── gaussian_manual_multi.py     # Used to collect and compute Gaussian color model
│
── data/
   └── src/
       ├── run_detection.py         # Main detection logic with confidence grid and Kalman tracking
       └── run_ab_sampling.py       # LAB data collection script


```
---


## 🔗 Repository Links

- [Main Project Directory](https://github.com/TongshuWu1/Tony_Swarm/tree/3298cbf0c3b08dae1506b79263e39706b3ddfb70/ComputerVision)
- [Detection Logic](https://github.com/TongshuWu1/Tony_Swarm/tree/main/ComputerVision/data/src)
- [LAB Color Gain Tools](https://github.com/TongshuWu1/Tony_Swarm/tree/main/ComputerVision/get_gain)


---
### 1. Calibrate Camera with `get_gain`

Before collecting LAB data or running detection, run the get_gains.py script to stabilize camera color balance:

you should get a list of gain values, but the one you need is

R,G,B = [aa,bb,cc]


Use this gain values in both sampling and detection scripts to lock color performance. At the top of the script, change the RGB value.

---

### 2. Collect LAB Color Data with `run_ab_sampling.py`

If you're working with a new balloon color, collect its LAB A/B values.

**Steps:**
1. Set `sampling_position = "center"` or `"left"` or `"right"` in the script.
2. Flash and run `run_ab_sampling.py`.
3. Hold balloon in front of camera with the grid on the balloon.
4. wait for around 30 second to get a consistant 2000 lines of data. 
5. Copy printed A/B values into a desinated color file (e.g, red_left.txt or red_center.txt). 
6. Use `gaussian_manual_multi.py` to calculate the Gaussian model (mean and covariance matrix).

---

### 4. Update Gaussian Info in Detection Script

In `run_detection.py`, replace the line that looks like:

```python
GAUSSIAN_INFO = '[44.609480198098309, 25.815678374] , [[0.05456789765434, -0.04865456789], [-0.04654567898765, 0.06457432789]]'
```

with the new values from your sampled data.

---

### 5. Run Real-Time Detection

Flash `run_detection.py` to the Nicla Vision. This script performs:

- Confidence grid scoring via LAB Gaussian model
- Grid cell decay and update
- Morphological cleaning
- Contour/Blob detection
- Target selection using area × confidence scoring
- Kalman filter-based motion prediction

It outputs:
- Cross marker (white): current estimated target location
- Circle (yellow): Kalman filter uncertainty radius

---

## Parameters You Can Tune

Located at the top of `run_detection.py`:

- `SENSITIVITY`: how strict the color match is. (2.7 is typical)
- `CONFIDENCE_THRESHOLD`: minimum score to count as positive.
- `DECAY`: how fast old confidence fades (0.7 = 30% loss/frame).
- `Q`, `R`: Kalman filter tuning for motion and measurement noise.
- `MAX_COV_TRACE`: if uncertainty grows beyond this, reset filter.

---

## Best Practices

- Use `get_gain.py` before every new environment (Especially during day night cycle time).
- Resample balloon color if lighting changes dramatically (but should be able to just use get_gains.py).
- Try to move balloons in different angle during sampling.
- Tune `SENSITIVITY` for different balloon types (red vs green).
- For noisy backgrounds, consider using neighbor filtering (future work).

