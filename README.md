# GAMBIT plotting tools

A repo for developing a collection of Python plotting tools for GAMBIT work. 


## Current working examples

### 2D profile likelihood plots

- Make a single 2D profile likelihood plot from the data in the hdf5 file `example_data/samples_run1.hdf5`
  ```terminal
  python example_2D_profile_like_hdf5.py
  ```

- Make multiple 2D profile likelihood plots in one go, combining the data from the hdf5 files `example_data/samples_run1.hdf5` and `example_data/samples_run2.hdf5`
  ```terminal
  python example_2D_profile_like_hdf5_multiple.py
  ```

- Make a single 2D profile likelihood plot from the data in the ascii file `example_data/samples.dat`
  ```terminal
  python example_2D_profile_like_ascii.py
  ```

### 2D posterior plots

- Make a single 2D posterior plot from the data in the hdf5 file `example_data/samples_multinest.hdf5`
  ```terminal
  python example_2D_posterior_hdf5.py
  ```
