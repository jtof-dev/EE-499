# wearable EEG research paper

- the main scripts in `eeg-data-analysis/` use `uv` and git submodules:

```bash
uv sync
git submodule update --init --recursive
```

## games

### `stroop/` 

- a simple local implementation of the stroop test, using `hjkl` keybinds (because I am a nerd!)
- logs basic info on the 5-minute session to a `metrics.csv`

### `typing/`

- a simple endless typing mode that resets on mistake
- logs basic info on the 5-minute session to a `metrics.csv`

### `n-back/`

- a local implementation of the n-back game (remember a feed of recently seen letters and record when the current matches the Nth back letter)
- logs basic info on the 5-minute session to a `metrics.csv`
- this is a potential option for a level 3 game, but I have not been able to get myself stressed enough to make it work
