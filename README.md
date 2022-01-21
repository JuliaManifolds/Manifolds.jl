<div align="center">
    <img src="https://github.com/JuliaManifolds/Manifolds.jl/blob/master/docs/src/assets/logo-text-readme.png" alt="Manifolds.jl" width="526">
</div>

| **Documentation** | **Source** | **Citation** |
|:-----------------:|:----------------------:|:------------:|
| [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliamanifolds.github.io/Manifolds.jl/stable/) | [![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle) | [![arXiv](https://img.shields.io/badge/arXiv%20CS.MS-2106.08777-blue.svg)](https://arxiv.org/abs/2106.08777) |
| [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliamanifolds.github.io/Manifolds.jl/latest/) | [![CI](https://github.com/JuliaManifolds/Manifolds.jl/workflows/CI/badge.svg)](https://github.com/JuliaManifolds/Manifolds.jl/actions?query=workflow%3ACI+branch%3Amaster) | [![DOI](https://zenodo.org/badge/190447542.svg)](https://zenodo.org/badge/latestdoi/190447542) |
| | [![codecov.io](http://codecov.io/github/JuliaManifolds/Manifolds.jl/coverage.svg?branch=master)](https://codecov.io/gh/JuliaManifolds/Manifolds.jl/) |


Package __Manifolds.jl__ aims to provide both a unified interface to define and
use manifolds as well as a library of manifolds to use for your projects.
This package is under development, and subject to changes as needed.

## Getting started

To install the package just type

```julia
] add Manifolds
```

Then you can directly start, for example to stop half way from the north pole on the [`Sphere`](https://juliamanifolds.github.io/Manifolds.jl/stable/manifolds/sphere.html) to a point on the equator, you can generate the [`shortest_geodesic`](https://juliamanifolds.github.io/Manifolds.jl/stable/interface.html#ManifoldsBase.shortest_geodesic-Tuple{AbstractManifold,Any,Any}).
It internally employs [`exp`](https://juliamanifolds.github.io/Manifolds.jl/latest/interface.html#Base.exp-Tuple{AbstractManifold,Any,Any}) and [`log`](https://juliamanifolds.github.io/Manifolds.jl/latest/interface.html#Base.log-Tuple{AbstractManifold,Any,Any}).

```julia
using Manifolds
M = Sphere(2)
γ = shortest_geodesic(M, [0., 0., 1.], [0., 1., 0.])
γ(0.5)
```

## Citation

If you use `Manifolds.jl` in your work, please cite the following

```biblatex
@online{2106.08777,
  Author = {Seth D. Axen and Mateusz Baran and Ronny Bergmann and Krzysztof Rzecki},
  Title = {Manifolds.jl: An Extensible {J}ulia Framework for Data Analysis on Manifolds},
  Year = {2021},
  Eprint = {2106.08777},
  Eprinttype = {arXiv},
}
```

To refer to a certain version we recommend to also cite for example

```biblatex
@software{manifoldsjl-zenodo-mostrecent,
  Author = {Seth D. Axen and Mateusz Baran and Ronny Bergmann},
  Title = {Manifolds.jl},
  Doi = {10.5281/ZENODO.4292129},
  Url = {https://zenodo.org/record/4292129},
  Publisher = {Zenodo},
  Year = {2021},
  Copyright = {MIT License}
}
```

for the most recent version or a corresponding version specific DOI, see [the list of all versions](https://zenodo.org/search?page=1&size=20&q=conceptrecid:%224292129%22&sort=-version&all_versions=True).
Note that both citations are in [BibLaTeX](https://ctan.org/pkg/biblatex) format.
