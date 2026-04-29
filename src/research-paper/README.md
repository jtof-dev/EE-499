# wearable EEG research paper

- the main scripts in `eeg-data-analysis/` use `uv` and git submodules:

```bash
uv sync
git submodule update --init --recursive
```

## games

- NOTE: the numbered streak feature was added on April 27th, and data taken before that did not have that element

### `stroop/` 

- a simple local implementation of the stroop test, using `hjkl` keybinds (because I am a nerd!)
- logs basic info on the 5-minute session to a `metrics.csv`

### `typing/`

- a simple endless typing mode that resets on mistake
- logs basic info on the 5-minute session to a `metrics.csv`

