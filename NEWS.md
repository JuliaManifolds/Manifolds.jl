# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.0] - 2023-mm-dd

### Added

- Vector bundles are generalized to fiber bundles.
- `RotationTranslationAction` is introduced.

### Changed

- Sizes of all manifolds can now be either encoded in type or stored in a field to avoid over-specialization.
  The default is set to store the size in type parameter, replicating the previous behavior.
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
- `HasLeftInvariantMetric`, `HasRightInvariantMetric` and `HasBiinvariantMetric` now explicitly refer to using default implementations of invariant metric functions. If you provide your own implementations, these traits should not be specified.

### Removed

- `ProductRepr` is removed; please use `ArrayPartition` instead.
- Default methods throwing "not implemented" `ErrorException` for some group-related operations. Standard `MethodError` is now thrown instead.
