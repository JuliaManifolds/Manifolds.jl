```@raw html
---
layout: home

hero:
  name: Manifolds.jl
  text: A Library of Riemannian Manifolds
  tagline: Efficient Numerical Differential Geometry
  actions:
    - theme: brand
      text: Get started
      link: tutorials/getstarted/index.html
  image:
    src: /logo.png            # primary image (light themes)
    dark: /logo-dark.png      # primary image (dark themes)
    alt: Manifolds.jl         # accessibility text

features:
  - icon:
        light: /logo-manifoldsbase.png
        dark: /logo-manifoldsbase-dark.png
        alt: ManifoldsBase.jl
        wrap: true
    title: Unified implementation
    details: All manifolds follow the interface from ManifoldsBase.jl.
    link: https://juliamanifolds.github.io/ManifoldsBase.jl/stable/
  - icon: 🧩
    title: Composable
    details: Components like a metric, the connection or different representations of points and tangent vectors are designed in a modular fashion. They can be reused and combined easily. Points and tangent vectors can be represented using arbitrary Julia types.
  - icon: 📚
    title: Well-documented and -tested
    details: All manifolds are documented – both their theoretical foundation and all numerical functionality. The theoretical background also refers to further literature.
  - icon: ⚡️
    title: Efficient
    link: https://github.com/JuliaManifolds/ManifoldsGPU.jl
    details: All methods provide also in-place evaluations. The package is compatible with running on a GPU when you use ManifoldsGPU.jl.
  - icon:
        light: /logo-liegroups.png
        dark: /logo-liegroups-dark.png
        alt: LieGroups.jl
        wrap: true
    title: Use with LieGroups.jl
    details: Together with a group operation, a manifold forms a Lie group. This package serves as base layer for LieGroups.jl.
    link: https://juliamanifolds.github.io/LieGroups.jl/stable/
  - icon:
        src: /logo-manopt.png
        alt: Manopt.jl
        wrap: true
    title: Use with Manopt.jl
    details: All manifolds from this package can be used as the domain of a function in the optimization algorithms of Manopt.jl
    link: https://manoptjl.org/stable/
---
```

```@docs
Manifolds.Manifolds
```

The manifolds are implemented using the interface for manifolds given in [`ManifoldsBase.jl`](@extref ManifoldsBase :doc:`index`).
You can use that interface to implement your own software on manifolds, such that all manifolds
based on that interface can be used within your code.

For more information, see the [About](misc/about.md) section.

## Getting started

To install the package just type

```julia
using Pkg; Pkg.add("Manifolds")
```

Then you can directly start, for example to stop half way from the north pole on the [`Sphere`](@ref) to a point on the the equator, you can generate the [`shortest_geodesic`](@extref `ManifoldsBase.shortest_geodesic-Tuple{AbstractManifold, Any, Any}`).
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

To refer to a specific version, it is recommended to cite, for example,

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
