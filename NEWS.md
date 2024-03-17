# `matplotlib_inline_labels` news

## v0.2.0 Release Notes

This release brings:
- performance enhancement, 20% to 50% faster than previous release
- a better handling of closed curves 

### Algorithm
---
- **Pre-processing** stage of creating label position candidates has been enhanced with:
  - a safeguard on the number of candidates via `maxpos` arg preventing cases of low `ppf`, or small label `fontsize` or high figure `figsize` or a conjunction of all, to lead to too many unnecessary position candidates
  - handling of line self-intersections
  - better handling of closed curved (position candidates added near the junction)
  - filtering of position candidates too close to any "distant" part of the same line (folds) or to any other line of the plot 
  - filtering of position candidates on a line part with a too high curvature for placing the label properly. Two performance equivalent options are available in the code `fast` and `precise` curvature estimation near the line chunks' edges. `fast` is used as default  
  *The gain on faster pre-processing is lost on processing, and the additional time used by the precise curvature estimation is compensated  by faster processing*
- **Processing** stage of finding the best label rotation for each label position candidate has been enhanced with:
  - limitation of the rotation tested angles, with the label's bounding box geometry, to pre-estimated best rotation +/- 2  x arctan(l/L)
  - limitation of the label separation computing to the label rotation with the smallest alignment error
  - creation of a flat list of position and rotation candidates structure for use in future post-processing algorithms

  The function giving the label's bounding box rotated and translated geometry, and its usage could still be optimized 

  The use of more discrete separation levels or a move to continuous separation levels still has to be explored to see if it leads to better label inline placement
- **Post-processing** stage has a draft algorithm using multi-objective optimization based on pymoo to enhance labels' position among themselves

### Code structure
---
- Data structures have been refactored for better readability and put in a dedicated file `datatypes.py`
- Functions of the main file `inline_labels.py` have been splitted in:
  - **drawing** (`drawing.py`): retrieval of matplotlib objects and creation of the visual debug figure
  - **geometries** (`geometries.py`): conversion of matplotlib objects to geometric objects
  - **pre-processing** (`preprocessing.py`): identification of label placement candidates on lines and filtering with line intersections and high curvatures
  - **processing** (`processing.py`): identification of label rotation and separation
  - **post-processing** (`postprocessing.py`): selection of the best candidate for each line

### Utils
---
- Timers are available to show the cumulated time of functions. Timers can be set as decorators (to be added or uncommented) on a function that could be optimized
- A timer decorator is available for the main function `add_inline_labels` for overall performance debug. It has to be uncommented to be used

### Misc
---
- Code has been reformatted with a max line length of 92 for better readability
- Github actions have been upgraded to last currently available versions
- Dropped python 3.11 compatibility to be able to use hashable slices. Maybe retintroduced later if requested, provided more advanced multi-objective optimization is used in post-processing
- Added a step in publish to pypi Github action to extract release notes from `NEWS.md`