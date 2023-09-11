# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.0] - 2023-mm-dd

### Added

- Vector bundles are generalized to fiber bundles.
- `RotationTranslationAction` is introduced.
- `DirectSumType` for vector bundles: `MultitangentBundle`, `MultitangentBundleFibers`, `MultitangentSpaceAtPoint`.

### Changed

- Sizes of all manifolds can now be either encoded in type or stored in a field to avoid over-specialization.
  The default is set to store the size in a field. To obtain the old behavior, pass the `parameter=:type` keyword
  argument to manifold constructor. Related changes:
  - Statically sized `SpecialEuclidean{N}` is now `SpecialEuclidean{TypeParameter{Tuple{N}}}`, whereas the type of special Euclidean group with field-stored size is `SpecialEuclidean{Tuple{Int}}`. Similar change applies to:
    - `CenteredMatrices{m,n}`,
    - `CholeskySpace{N}`,
    - `Elliptope{N,K}`,
    - `Euclidean`,
    - `FixedRankMatrices{m,n,k}`,
    - `KendallsPreShapeSpace{n,k}`,
    - `KendallsShapeSpace{n,k}`,
    - `GeneralLinear{n}`,
    - `GeneralUnitaryMultiplicationGroup{n}`,
    - `GeneralizedGrassmann{n,k}`,
    - `GeneralizedStiefel{n,k}`,
    - `Grassmann{n,k}`,
    - `Heisenberg{n}`,
    - `Hyperbolic{n}`,
    - `MultinomialMatrices{N,M}`,
    - `MultinomialDoublyStochastic{n}`,
    - `MultinomialSymmetric{n}`,
    - `Orthogonal{n}`,
    - `ProbabilitySimplex{n}`,
    - `SpecialOrthogonal{n}`,
    - `SpecialUnitary{n}`,
    - `SpecialEuclideanManifold{n}`,
    - `SphereSymmetricMatrices{N}`,
    - `Stiefel{n,k}`,
    - `SymmetricMatrices{N}`,
    - `SymmetricPositiveDefinite{n}`,
    - `SymmetricPositiveSemidefiniteFixedRank{n,k}`,
    - `TranslationGroup`,
    - `Tucker`.
  
  For example

  ```{julia}
  function Base.show(io::IO, ::SpecialEuclidean{n}) where {n}
      return print(io, "SpecialEuclidean($(n))")
  end
  ```

  needs to be replaced with

  ```{julia}
  function Base.show(io::IO, ::SpecialEuclidean{TypeParameter{Tuple{n}}}) where {n}
      return print(io, "SpecialEuclidean($(n); parameter=:type)")
  end
  ```

  for statically-sized groups and

  ```{julia}
  function Base.show(io::IO, G::SpecialEuclidean{Tuple{Int}})
    n = get_n(G)
    return print(io, "SpecialEuclidean($(n))")
  end
  ```

  for groups with size stored in field. Alternatively, you can use a single generic method like this:
  
  ```{julia}
  function Base.show(io::IO, G::SpecialEuclidean{T}) where {T}
    n = get_n(G)
    if T <: TypeParameter
        return print(io, "SpecialEuclidean($(n); parameter=:type)")
    else
        return print(io, "SpecialEuclidean($(n))")
    end
  end
  ```

- Argument order for type aliases `RotationActionOnVector` and `RotationTranslationActionOnVector`: most often dispatched on argument is now first.

### Removed

- `ProductRepr` is removed; please use `ArrayPartition` instead.
- Default methods throwing "not implemented" `ErrorException` for some group-related operations.
