# Probabilistic SLAM in Gen

## **Localization Tutorial**

### **Notebooks**

The main notebooks with the actual tutorials can be found in `notebooks/v2/`:
- [Localization Tutorial - Part 1.ipynb](notebooks/v2/51%20-%20Localization%20Tutorial%20-%20Part%201.ipynb)
- [Localization Turorial - Part 2.ipynb](notebooks/v2/52%20-%20Localization%20Tutorial%20-%20Part%202.ipynb)
- [Localization Turorial - Part 3.ipynb](notebooks/v2/53%20-%20Localization%20Tutorial%20-%20Part%203.ipynb)

Overview of all relevant notebooks and directories:
```
notebooks/
│
│   00 - My Utils
│   00 - Cuda Utils
│   01 - Geometry - Primitives and Raycaster.ipynb
│   11 - Pose.ipynb
│   31b - CUDA Raycaster - Line Map.ipynb
│
├───src/  (...source files compiled from nb's)│
├───data/ (...)
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
    ├───src/   (...source files compiled from nb's)
    └───_imgs/ (...where we store generated figures)
```

### **Setup**

- Run `setup.jl` to install the packages in the `REQUIRE` file. This just calls `Pkg.add` on each of the entries in `REQUIRE`. 
- There is a `Project.toml` as well.
- You might want to create directories `notebooks/_imgs`, and `notebooks/v2/_imgs`. Note that in this repo we don't track files or folders with a leading underscore `_ignore_this.txt`; see the `.gitignore`.
- Clone https://github.com/mirkoklukas/Gen-Distribution-Zoo and add its source to the load path or set the environment variable "probcomp" to the folder you cloned this repo to. In the notebooks I call: `push!(LOAD_PATH, ENV["probcomp"]*"/Gen-Distribution-Zoo/src")`

## **Notes**

In this repo we don't track files or folders with a leading underscore `_ignore_this.txt`; see the `.gitignore`.

## **Notation and naming conventions**

I usually try to keep variable naming concise and closer to the conventions found in mathematical papers, rather than those used in software engineering. I find that adopting this approach makes it easier to comprehend research code. However, it is essential to provide additional notes or documentation to provide context for interpreting variable names.

```julia
    x              = rand(2)  # Preferred
    agent_position = rand(2)  # Sometimes OK as well

    ỹ = sliding_windows(y, 10, 1)                      # Preferred
    sliding_windows_over_y = sliding_windows(y, 10, 1) # Hard to parse
    
```

In line with this practice, I make an effort to indicate CUDA arrays by appending a trailing underscore. For example, if we have an array `x`, its CUDA version would be denoted as `x_`:

```julia
    x  = rand(100,100)
    x_ = CuArray(x)
```

Following a Python-inspired convention, I use leading underscores for contants or varibales that are not supposed to change. This allows me to reuse descriptive names without conflicting with constant values. Consequently, I can employ these informative names directly in the code that follows, ensuring clarity and readability.

```julia
    ps = [ p+u for (p, u) in zip(_ps, _us) ]
```