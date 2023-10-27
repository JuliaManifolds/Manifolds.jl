# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.2] - 2023-10-27

### Added

- `rand(G; vector_at=Identity(G))` now works for translation, special orthogonal and special Euclidean groups `G` (issue [#665](https://github.com/JuliaManifolds/Manifolds.jl/issues/665)).
- `get_embedding` now works for `GeneralUnitaryMultiplicationGroup`.
- Github action that checks for NEWS.md changes.

## [0.9.1] - 2023-10-25

### Added

- a new retraction and its inverse for the fixed Rank Manifolds, the orthographic retraction.

## [0.9.0] - 2023-10-24

### Added

- Vector bundles are generalized to fiber bundles. Old `BundleFibers` functionality was reworked to better match mathematical abstractions. Fiber bundle functionality is experimental and minor changes may happen without a breaking release, with the exception of `TangentBundle` which is considered to be stable.
- `RotationTranslationAction` is introduced.

### Changed

- Sizes of all manifolds can now be either encoded in type or stored in a field to avoid over-specialization.
  The default is set to store the size in type parameter (except for `PowerManifold` and its variants), replicating the previous behavior.
  For field storage, pass the `parameter=:field` keyword argument to manifold constructor.
  For example statically sized `CenteredMatrices{m,n}` is now `CenteredMatrices{TypeParameter{Tuple{m,n}}}`, whereas the type of special Euclidean group with field-stored size is `CenteredMatrices{Tuple{Int,Int}}`. Similar change applies to:
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
  - `PowerManifold`,
  - `PositiveArrays`,
  - `PositiveMatrices`,
  - `PositiveNumbers`,
  - `ProbabilitySimplex{n}`,
  - `SPDFixedDeterminant{n}`,
  - `SpecialLinear{n}`,
  - `SpecialOrthogonal{n}`,
  - `SpecialUnitary{n}`,
  - `SpecialEuclidean{n}`,
  - `SpecialEuclideanManifold{n}`,
  - `Spectrahedron{n,k}`,
  - `SphereSymmetricMatrices{N}`,
  - `Stiefel{n,k}`,
  - `SymmetricMatrices{N}`,
  - `SymmetricPositiveDefinite{n}`,
  - `SymmetricPositiveSemidefiniteFixedRank{n,k}`,
  - `Symplectic{n}`,
  - `SymplecticStiefel{n,k}`,
  - `TranslationGroup`,
  - `Tucker`.

  For example

  ```{julia}
  function Base.show(io::IO, ::CenteredMatrices{m,n}) where {m,n}
      return print(io, "CenteredMatrices($m, $n)")
  end
  ```

  needs to be replaced with

  ```{julia}
  function Base.show(io::IO, ::CenteredMatrices{TypeParameter{Tuple{m,n}}}) where {m,n}
      return print(io, "CenteredMatrices($m, $n)")
  end
  ```

  for statically-sized groups and

  ```{julia}
  function Base.show(io::IO, M::CenteredMatrices{Tuple{Int,Int}})
      m, n = get_parameter(M.size)
      return print(io, "CenteredMatrices($m, $n; parameter=:field)")
  end
  ```

  for groups with size stored in field. Alternatively, you can use a single generic method like this:

  ```{julia}
  function Base.show(io::IO, M::CenteredMatrices{T}) where {T}
      m, n = get_parameter(M)
      if T <: TypeParameter
          return print(io, "CenteredMatrices($m, $n)")
      else
          return print(io, "CenteredMatrices($m, $n; parameter=:field)")
      end
  end
  ```

- Argument order for type aliases `RotationActionOnVector` and `RotationTranslationActionOnVector`: most often dispatched on argument is now first.
- A more consistent handling of action direction was introduced. 4-valued `ActionDirection` was split into 2-valued `ActionDirection` (either left or right action) and `GroupActionSide` (action acting from the left or right side). See [https://github.com/JuliaManifolds/Manifolds.jl/issues/637](https://github.com/JuliaManifolds/Manifolds.jl/issues/637) for a design discussion.

### Removed

- `ProductRepr` is removed; please use `ArrayPartition` instead.
- Default methods throwing "not implemented" `ErrorException` for some group-related operations. Standard `MethodError` is now thrown instead.
- `LinearAffineMetric` was deprecated in a previous release and the symbol is now removed.
  Please use `AffineInvariantMetric` instead.
