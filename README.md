<div align="center">
    <img src="https://github.com/JuliaManifolds/Manifolds.jl/blob/master/docs/src/assets/logo-text-readme.png" alt="Manifolds.jl" width="526">
</div>

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliamanifolds.github.io/Manifolds.jl/stable/)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliamanifolds.github.io/Manifolds.jl/latest/)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![CI](https://github.com/JuliaManifolds/Manifolds.jl/workflows/CI/badge.svg)](https://github.com/JuliaManifolds/Manifolds.jl/actions?query=workflow%3ACI+branch%3Amaster)
[![codecov.io](http://codecov.io/github/JuliaManifolds/Manifolds.jl/coverage.svg?branch=master)](https://codecov.io/gh/JuliaManifolds/Manifolds.jl/)
[![DOI](https://zenodo.org/badge/190447542.svg)](https://zenodo.org/badge/latestdoi/190447542)

Package __Manifolds.jl__ aims to provide both a unified interface to define and
use manifolds as well as a library of manifolds to use for your projects.
This package is under development, and subject to changes as needed.

## Getting started

To install the package just type

```julia
] add Manifolds
```

Then you can directly start, for example to stop half way from the north pole on the [`Sphere`](https://juliamanifolds.github.io/Manifolds.jl/stable/manifolds/sphere.html) to a point on the the equator, you can generate the [`shortest_geodesic`](https://juliamanifolds.github.io/Manifolds.jl/stable/interface.html#ManifoldsBase.shortest_geodesic-Tuple{AbstractManifold,Any,Any}).
It internally employs [`exp`](https://juliamanifolds.github.io/Manifolds.jl/latest/interface.html#Base.exp-Tuple{AbstractManifold,Any,Any}) and [`log`](https://juliamanifolds.github.io/Manifolds.jl/latest/interface.html#Base.log-Tuple{AbstractManifold,Any,Any}).

```julia
using Manifolds
M = Sphere(2)
γ = shortest_geodesic(M, [0., 0., 1.], [0., 1., 0.])
γ(0.5)
```
