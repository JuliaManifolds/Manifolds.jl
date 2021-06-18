# Manifolds

The package __Manifolds__ aims to provide a library of manifolds to be used within your project.
The implemented manifolds are accompanied by their mathematical formulae.

The manifolds are implemented using the interface for manifolds given in [`ManifoldsBase.jl`](interface.md).
You can use that interface to implement your own software on manifolds, such that all manifolds
based on that interface can be used within your code.

For more information, see the [About](misc/about.md) section.

## Getting started

To install the package just type

```julia
] add Manifolds
```

Then you can directly start, for example to stop half way from the north pole on the [`Sphere`](@ref) to a point on the the equator, you can generate the [`shortest_geodesic`](@ref).
It internally employs [`log`](@ref log(::Sphere,::Any,::Any)) and [`exp`](@ref exp(::Sphere,::Any,::Any)).

```@example
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
Title = {Manifolds.jl: An Extensible Julia Framework for Data Analysis on Manifolds},
Year = {2021},
Eprint = {2106.08777},
Eprinttype = {arXiv},
}
```

and to refer to a certain version we recommend to also cite for example

```biblatex
@softawre{manifoldsjl-zenodo-mostrecent,
  Author = {Set D. Axen and Mateusz Baran and Ronny Bergmann},
  Title = {Manifolds.jl},
  Doi = {10.5281/ZENODO.4292129},
  Url = {https://zenodo.org/record/4292129},
  Publisher = {Zenodo},
  Year = {2021},
  Copyright = {MIT License}
}
```

for the most recent version or a corresponding version specific DOI, see [the list of all versions](https://zenodo.org/search?page=1&size=20&q=conceptrecid:%224292129%22&sort=-version&all_versions=True).
