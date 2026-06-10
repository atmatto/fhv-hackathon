# CVD Death-Risk Model — Training & Dev

Estimates a person's risk of dying from cardiovascular disease over time
from ordinary health inputs, and serves it through a small web app. Trained on
U.S. NHANES survey data linked to National Death Index mortality follow-up.

## What it predicts

The outcome is cause-specific cardiovascular death (NCHS leading-cause
recode "Diseases of heart"), with follow-up in person-months from the exam
date. So it estimates time to CVD death, not "time until you first
develop a heart problem."

It leads with horizon risk — "X% chance of CVD death within 1 / 5 / 10
years" — rather than a single death date, because for most people the risk
never reaches 50% inside the follow-up window, so a date would be invented.

The number it reports is a cumulative incidence function (CIF): the chance
of dying of CVD by year t given you might die of something else first.
That competing-risks framing is what makes the percentage honest (see below).

## Run

```
uv run build_real_model.py
```
