# Probabilistic SLAM in Gen

## **Localization Tutorial**

### Notebooks

The main notebooks with the actual tutorials can be found in `notebooks/v2/`:
- [Localization Tutorial - Part 1.ipynb](notebooks/v2/51%20-%20Localization%20Tutorial%20-%20Part%201.ipynb)
- [Localization Turorial - Part 2.ipynb](notebooks/v2/52%20-%20Localization%20Tutorial%20-%20Part%202.ipynb)
- [Localization Turorial - Part 3.ipynb](notebooks/v2/53%20-%20Localization%20Tutorial%20-%20Part%203.ipynb)

Overview of all relevant notebooks:
```
notebooks/
│
│   00 - My Utils
│   00 - Cuda Utils
│   01 - Geometry - Primitives and Raycaster.ipynb
│   11 - Pose.ipynb
│   31b - CUDA Raycaster - Line Map.ipynb
│
├───src/ (...source files compiled from nb's)
│
└───v2/
    │
    │   01 - HouseExpo Data.ipynb
    │   02 - CSAIL Data.ipynb
    │   13 - 2dp3 Sensor Distribution.ipynb
    │   (...)
    │   51 - Localization Tutorial - Part 1
    │   52 - Localization Tutorial - Part 2
    │   53 - Localization Tutorial - Part 3
    │
    └───src/ (...source files compiled from nb's)
```

### Requirements

- Run `setup.jl` to install the packages in the `REQUIRE` file. This just calls `Pkg.add` on each of the entries in `REQUIRE`. 
- There is a `Project.toml` as well.
- Clone https://github.com/mirkoklukas/Gen-Distribution-Zoo and add its source to the load path or set the environment variable "probcomp" to the folder you cloned this repo to. In the notebooks I call: `push!(LOAD_PATH, ENV["probcomp"]*"/Gen-Distribution-Zoo/src")`




