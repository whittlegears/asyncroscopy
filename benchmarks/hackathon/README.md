# Hackathon task: when does a smarter controller beat a raster?

You get a simulated STEM, a sample you know nothing about and 700 seconds of instrument time. Work out what the sample is made of and where. It runs on a laptop: no microscope, no servers, no GPU.

```bash
uv sync --extra agent
uv run python -m benchmarks.hackathon                                      # the three baselines on three samples
uv run python -m benchmarks.hackathon --controller my_team.py:controller   # yours beside them
```

Scores print as a table. Pictures of every run go to `outputs/hackathon/`: the survey as the microscope imaged it, the map the controller answered and the true map.

## The sample

An 8 um square of carbon film with about 6000 nanoparticles of Au, Pt, Fe and Pd (`configs/digital_twin_heterogeneous.yaml`). It is cut into 40 regions. Each region has its own particle density, film thickness, element mix and particle size, and some regions are holes.

One element, the **marker**, sets the particle size: the larger its share of a region's particles, the smaller they are. Which element it is changes from sample to sample. Images show sizes and spectra show elements, so you need both to find it.

## What you control

A controller is a function that takes a `Microscope` and returns an `Answer` (`task.py`):

| Call | What it does | Costs about |
| :-- | :-- | :-- |
| `microscope.move(x, y)` | Moves the stage (metres, origin at the centre). It does not land exactly where asked, and the sample drifts for some seconds afterwards. | 1.5 s |
| `microscope.image()` | One 256 x 256 HAADF image, 400 nm wide, where the stage is. | 0.2 s |
| `microscope.spectrum(column, row)` | One EDS spectrum with the beam on that pixel of the last image. | 0.5 s |
| `microscope.autofocus()` | The instrument's autofocus; returns the defocus it found. | 1.6 s |
| `microscope.set_defocus(d)` | Sets the defocus you say. | nothing |

`microscope.remaining_s` says what is left. When the budget is spent every call raises `BudgetSpent`: catch it and answer with what you have.

The specimen is not flat and the focus drifts, so the focus that was right at one place is wrong at the next. Spectra taken long after their image miss the particle, because the sample has drifted.

## What you hand back

```python
Answer(
    marker_element="Pt",            # which element sets the size
    size_ratio=0.4,                 # radius where every particle is the marker / radius where none is
    share_map=...,                  # 16 x 16 floats: the marker's share of the particles in each cell
    hole_map=...,                   # 16 x 16 booleans: which cells are holes
)
```

Maps are indexed `[row, column]` with rows along +y, like an image. `task.cell_of(x, y)` gives the cell of a stage point.

## How it is scored

`score.py`, out of 100:

| Part | Points | For |
| :-- | --: | :-- |
| Marker | 30 | Naming the right element. |
| Rule | 30 | How close `size_ratio` is to the truth (about 0.33). None with the wrong marker, none for "size does not depend on the element". |
| Map | 40 | The share map and hole map, cell by cell, as the improvement over answering the sample's mean share everywhere with no holes. |

The truth comes from the twin and is never shown to a controller. Report the mean over the samples you were given; the final ranking uses samples (seeds) nobody has seen.

## The baselines

`baselines.py`. All three image every map cell in a serpentine and take three spectra on random particles of each image. They differ only in how they focus.

| Baseline | How it focuses |
| :-- | :-- |
| `raster` | Once at the start. |
| `periodic` | Every fourth image. |
| `surface` | Remembers what each autofocus run found, fits a smooth surface through those places and sets the predicted defocus for free; it runs autofocus only where it has no measurement nearby. |

Mean over samples 101 to 105 (reproduce with `uv run python -m benchmarks.hackathon --seeds 101 102 103 104 105`):

| Controller | Score | Marker | Rule | Map | Images | Autofocus runs |
| :-- | --: | --: | --: | --: | --: | --: |
| raster | 69.0 | 30.0 | 20.8 | 18.1 | 256 | 1 |
| periodic | 72.0 | 30.0 | 24.0 | 18.0 | 239 | 60 |
| surface | 73.7 | 30.0 | 25.3 | 18.4 | 244 | 52 |

What this says: focusing well is worth a few points, and a model of the focus surface gets them with fewer autofocus runs. It does not say much more. The three are within five points of each other, on a single sample the order can swap, and none of them gets half the map points.

## Where the points are

None of the baselines chooses anything about where to look.

- **The map** is worth 40 and the baselines get 18. They spend the same time on a hole, an empty region and a crowded one, and three spectra per image say little about one cell.
- **The rule** needs regions with very different marker shares. Finding two such regions early and measuring them well beats measuring everything a little.
- **Spectra** cost more than twice an image. Where is one worth it?
- **Position.** Images are drawn where the stage says it is, which is off by up to a few hundred nanometres.

`analysis.py` has the simple particle finder, hole test and element identification the baselines use. Use them, improve them or replace them.

## Files

| File | What it is |
| :-- | :-- |
| `task.py` | The task: the microscope a controller drives, the answer it returns, the budget. |
| `score.py` | The scorer. |
| `analysis.py` | Image and spectrum analysis for the baselines. |
| `baselines.py` | The three baselines. |
| `__main__.py` | Runs controllers, prints the table, draws the pictures. |
