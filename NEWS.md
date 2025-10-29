# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.11.2] unreleased

### Fixed

* fixed the `default_retraction_method` for `AbstractSphere` to be the stabilized exponential map to avoid accumulations of rounding errors
  when the tangent vector is not exactly tangent.
* fixed the contributing.md to mention runic as the code formatter.

## [0.11.1] 2025-10-24

### Fixed

* fix a few small errors regarding the forwarding of `retract`, `retract_fused`, and `inverse_retract` when using a `MetricManifold.jl` (cf (#823))
* document why we kept that `vector_transport_direction` and `vector_transport_to` are not automatically forwarded in `MetricManifold.jl`

## [0.11.0] 2025-10-02

### Added

* a function `metric(M)` analogous to `embedding(M)` and `connection(M)` that returns the metric of the manifold `M`.
* a `DefaultMetric()` type to specify that the manifold uses some specific metric, that is not further named.

### Changed

* Switch to using [Runic.jl](https://github.com/fredrikekre/Runic.jl) as code formatter
* refactored all manifolds to use a parameter order of `{𝔽, T}` where `𝔽` is the number type and `T` the type of size parameter (either a `TypeParameter` or a `Tuple`).
  this changed internally
  * `ArraySphere{T<:Tuple,𝔽}` to `ArraySphere{𝔽, T<:Tuple}`
  * `CenteredMatrices{T, 𝔽}` to `CenteredMatrices{𝔽, T}`
  * `Euclidean{T, 𝔽}` to `Euclidean{𝔽, T}`
  * `FixedRankMatrices{T, 𝔽}` to `FixedRankMatrices{𝔽, T}`
  * `GeneralizedGrassmann{T, 𝔽, B}` to `GeneralizedGrassmann{𝔽, T, B}`
  * `GeneralizedStiefel{T, 𝔽, B}` to `GeneralizedStiefel{𝔽, T, B}`
  * `Grassmann{T, 𝔽, B}` to `Grassmann{𝔽, T, B}`
  * `GeneralUnitaryMatrices{T, 𝔽, S}` to `GeneralUnitaryMatrices{𝔽, T, S}`
  * `GraphManifold{G <: AbstractGraph, 𝔽, TM, T <: GraphManifoldType}` to `GraphManifold{𝔽, G <: AbstractGraph, TM, T <: GraphManifoldType}`
  * `Grassmann{T, 𝔽}` to `Grassmann{𝔽, T}`
  * `HamiltonianMatrices{T, 𝔽}` to `HamiltonianMatrices{𝔽, T}`
  * `Oblique{T, 𝔽, S}` to `Oblique{𝔽, T, S}`
  * `SkewHermitianMatrices{T, 𝔽}` to `SkewHermitianMatrices{𝔽, T}`
  * `Sphere{T, 𝔽}` to `Sphere{𝔽, T}`
  * `SphereSymmetricMatrices{T, 𝔽}` to `SphereSymmetricMatrices{𝔽, T}`
  * `Stiefel{T, 𝔽}` to `Stiefel{𝔽, T}`
  * `SymmetricMatrices{T, 𝔽}` to `SymmetricMatrices{𝔽, T}`
  * `SymplecticGrassmann{T, 𝔽}` to `SymplecticGrassmann{𝔽, T}`
  * `SymplecticStiefel{T, 𝔽}` to `SymplecticStiefel{𝔽, T}`
  * `Tucker{T, D, 𝔽}` to `Tucker{𝔽, T, D}`
  * `UnitaryMatrices{T, 𝔽}` to `UnitaryMatrices{𝔽, T}`
* the functions `canoncial_project`, `differential_canonical_project`, `horizontal_lift`, `horizontal_component`, `get_total_space`, and `vertical_component` have been moved upstream to `ManifoldsBase.jl`


### Removed

* the trait system that was present additionally to the wrapper types for metric, embedding and connections. Those were moved to `default_X` functions and the dispatch was adapted to still obtain the same logic/behaviour.
* the `QuotientManifold` type, since it was not really used and its proper definition would require functionality from [LieGroups.jl](https://juliamanifolds.github.io/LieGroups.jl/stable/)
* all functions related to the `GroupManifold`s type and corresponding traits using the old trait system have been removed here after being deprecated for a while now.
  These have been redesigned and introduced in the new package [LieGroups.jl](https://juliamanifolds.github.io/LieGroups.jl/stable/), see their [How to transition from `GroupManifold`s tutorial](https://juliamanifolds.github.io/LieGroups.jl/stable/tutorials/transition/) for all details.
* all deprecated tangent vector types that had `TVector` in their name
  * instead of `HyperboloidTVector`use`HyperboloidTangentVector`
  * instead of `OrthogonalTVector` use`OrthogonalTangentVector`
  * instead of `PoincareBallTVector` use `PoincareBallTangentVector`
  * instead of `PoincareHalfSpaceTVector` use `PoincareHalfSpaceTangentVector`
  * instead of `ProjectorTVector` use `ProjectorTangentVector`
  * instead of `StiefelTVector` use `StiefelTangentVector`
  * instead of `TuckerTVector` use `TuckerTangentVector`
  * instead of `UMVTVector` use `UMVTangentVector`
* The `QuotientManifold` type has been removed, the API for general (implicit) quotient manifolds is now already in `ManifoldsBase.jl`.
* The  `retract_exp_ode!` and `retract_exp_ode_fused!` functions were removed. The cases in which they were correct are covered by `solve_chart_exp_ode`, and it was too easy to use `retract_exp_ode!` incorrectly.

## [0.10.23] 2025-07-19

### Fixed

* fix a small bug in the point checks of general unitary matrices.

## [0.10.22] 2025-06-25

### Fixed

* Support for `ForwardDiff.jl` v1

## [0.10.21] 2025-06-24

### Added

* Support for `ForwardDiff.jl` v1

## [0.10.20] 2025-06-16

### Fixed

* Fix in-place computation of `exp!` for `GeneralLinear`.

## [0.10.19] 2025-06-14

### Added

* `riemannian_gradient` for the `GeneralLinear` manifold
* `riemannian_gradient` method for `ProductManifold`.

### Fixed

* Fix the projection and embedding of tangent vectors on `GeneralLinear`.

## [0.10.18] 2025-05-29

### Fixed

* Fix the supertype of `PoincareBallTangentVector` to be `AbstractTangentVector`
* Fix the supertype of `StiefelTangentVector` to be `AbstractTangentVector`
* Fix `riemannian_gradient` for fixed rank matrices, which did not work due to a small bug in the default fallback and a missing metric specification.

## [0.10.17] 2025-04-21

### Changed

* deprecate `GroupManifold`s and its concrete subtypes as well as all functions related to Lie groups.
  They can now be found in the new package [LieGroups.jl](https://juliamanifolds.github.io/LieGroups.jl/stable/), see their [How to transition from `GroupManifold`s tutorial](https://juliamanifolds.github.io/LieGroups.jl/stable/tutorials/transition/) for all details.

Note that while it is currently not so easy to use `Manifolds.jl` and `LieGroups.jl` together due to the
common definitions that are here now deprecated, it might still take a reasonable time to do a breaking
release here, since we do not have a very good reason to yet.

## [0.10.16] 2025-04-08

### Changed

* Added all `[compat]` entries also for the `docs/` and `tutorials/` environments
* `get_vector` on `Circle` no longer returns `SArray` when no static arrays are passed as arguments.

### Fixed

* a copy pase error in the new determinant one manifold from the last release.

## [0.10.15] 2025-03-28

### Added

* a `DeterminantOneMatrices` manifold of matrices of determinant one.

### Changed

* the following internal types were renamed since their super type also uses the singular
  * `AbsoluteDeterminantOneMatrices` to `AbsoluteDeterminantOneMatrixType`
  * `DeterminantOneMatrices` to `DeterminantOneMatrixType`

### Fixed

* fix `rand!` to also work on the `Circle(ℂ)`

## [0.10.14] - 2025-02-18

### Changed

* Introduced new implementation of parallel transport on `Rotations(3)` based on Rodrigues' rotation
formula.

## [0.10.13] - 2025-02-10

### Changed

* Bumped dependency of ManifoldsBase.jl to 1.0, split `exp` into `exp` (without optional argument `t`) and `exp_fused` (with argument `t`) and similarly `retract` to `retract` and `retract_fused`.
* ManifoldsBase.jl 1.0 also moved from `TVector` to `TangentVector`s in type names.
  The following names are adapted
  * Renamed `HyperboloidTVector` (now deprecated) to `HyperboloidTangentVector`
  * Renamed `OrthogonalTVector` (now deprecated) to `OrthogonalTangentVector`
  * Renamed `PoincareBallTVector` (now deprecated) to `PoincareBallTangentVector`
  * Renamed `PoincareHalfSpaceTVector` (now deprecated) to `PoincareHalfSpaceTangentVector`
  * Renamed `ProjectorTVector` (now deprecated) to `ProjectorTangentVector`
  * Renamed `StiefelTVector` (now deprecated) to `StiefelTangentVector`
  * Renamed `TuckerTVector` (now deprecated) to `TuckerTangentVector`
  * Renamed `UMVTVector` (now deprecated) to `UMVTangentVector`
* The internal access `array_value` is now called `internal_value`, compare to its renaming in `ManifoldsBase`

## [0.10.12] - 2025-01-10

### Added

* Orthonormal bases for `CholeskySpace` and `LogCholesky` metric for `SymmetricPositiveDefinite`.
* `rand` for `CholeskySpace`.

### Changed

* Improved performance of selected `get_vector` and `get_coordinates` methods for complex `Euclidean` manifold.

## [0.10.11] - 2025-01-02

### Added

* Bases and rand for `HeisenbergMatrices` and `InvertibleMatrices`.

## [0.10.10] - 2024-12-20

### Added

* the `Segre` manifold
* the `WarpedMetric` for the `Segre`manifold

## [0.10.9] - 2024-12-16

### Added

* the `Segre` manifold
* the `WarpedMetric` for the `Segre`manifold
* The manifold `HeisenbergMatrices` as the underlying manifold of `HeisenbergGroup`.

### Changed

* `about.md` now also lists contributors of manifolds and a very short history of the package.

## [0.10.8] – 2024-11-27

### Changed

* Some methods related to `get_vector` for `GeneralUnitaryMatrices` now have `AbstractVector` upper bound for coefficients.
* Minimum Julia version is now 1.10 (the LTS which replaced 1.6)
* The dependency ManifoldDiff.jl has been upgraded from v0.3 to v0.4, to bring compatibility with DifferentiationInterface.jl.

## [0.10.7] – 2024-11-16

### Added

* `adjoint_matrix` for Lie groups, with optimized implementations for SO(2), SO(3), SE(2) and SE(3).

## [0.10.6] – 2024-11-06

### Added

* Two new actions: `ComplexPlanarRotation`, `QuaternionRotation`.
* New function `quaternion_rotation_matrix` for converting quaternions to rotation matrices.
* `make.jl` script now has more command line arguments, for example `--exclude-tutorials` when you do not want to build the tutorials but still look at the docs. See `make.jl --help` for more information.

## [0.10.5] – 2024-10-24

### Added

* the manifold `InvertibleMatrices` of invertible matrices

### Changed

* rewrote the `CONTRIBUTING.md` and adapt it to today's links and references.

## [0.10.4] - 2024-10-20

### Added

* `uniform_distribution` now has an error hint explaining what has to be done to make it work.
* `Euclidean` now follows the new `has_components` function from `ManifoldsBase.jl` (0.15.18)
  and can handle also the `r`-norms now.
- Union type `MatrixGroup`
- Columnwise group action with arbitrary matrix groups
- `uniform_distribution` now has an error hint explaining what has to be done to make it work.
- `lie_bracket` is exactly zero on orthogonal Lie algebra in 2D

## [0.10.3] - 2024-10-04

### Changed

* **Mildly breaking**: the number system parameter now corresponds to the coefficients standing in front of basis vectors in a linear combination instead of components of a vector. For example, `DefaultOrthonormalBasis() == DefaultOrthonormalBasis(ℝ)` of `Euclidean(3, field=ℂ)` now has 6 vectors, and `DefaultOrthonormalBasis(ℂ)` of the same manifold has 3 basis vectors.

### Fixed

* Fixed `solve_exp_ode` only returning the starting position ([#744](https://github.com/JuliaManifolds/Manifolds.jl/issues/744))
* Fixed documentation of `solve_exp_ode` function signature ([#740](https://github.com/JuliaManifolds/Manifolds.jl/issues/740))

## [0.10.2] - 2024-09-24

### Added

* `GroupManifold` can now be called with two arguments, the third one defaulting to `LeftInvariantRepresentation`.

### Changed

* fixes a few typographical errors.

## [0.10.1] – 2024-08-29

### Changed

* `identity_element` on `ProductManifold` without `RecursiveArrayTools.jl` now prints a useful error message.

## [0.10.0] – 2024-08-24

### Changed

* `Distributions.jl`, `RecursiveArrayTools.jl` and `HybridArrays.jl` were moved to weak dependencies to reduce load time and improve extensibility.
* `translate_diff`, `inv_diff` and thus `apply_diff_group`, are available for all the groups with invariant tangent vector storage.
* `SpecialEuclidean` group now has a different default tangent vector representation, the left-invariant one; to get the old representation pass `vectors=HybridTangentRepresentation()` to the constructor of `SpecialEuclidean`.
* `adjoint_action` takes a direction argument; by default it is `LeftAction`.
* `adjoint_action!` is the necessary method to implement in groups with left-invariant tangent vector representation.
* Fixed a few typos in the doc string of the SPD fixed determinant description.
* Random point on the `MultinomialSymmetricPositiveDefinite` manifold was improved to make it more robust.

### Added

* Introduced `exp_inv` and `log_inv` based on `exp_lie` and `log_lie`. They are invariant to the group operation.
* A tutorial about usage of group-related functionality.

### Removed

* Deprecated bindings:
  * `ExtrinsicEstimation()` (should be replaced with `ExtrinsicEstimation(EfficientEstimator())`),
  * `Symplectic` (renamed to `SymplecticMatrices`),
  * `SymplecticMatrix` (renamed to `SymplecticElement`).
  * `AbstractEstimationMethod` (renamed to `AbstractApproximationMethod`).
  * `VectorBundleVectorTransport` (renamed to `FiberBundleProductVectorTransport`).
  * `rand` on `SymplecticMatrices` and `SymplecticStiefel` no longer accepts `hamiltonian_norm` as an alias for `σ`.
  * `mean!` and `median!` no longer accept `extrinsic_method` (should be replaced with `e = ExtrinsicEstimation(extrinsic_method)`).
* As a result of making `Distributions.jl` and `RecursiveArrayTools.jl` weak dependencies the following symbols are no longer exported from `Manifolds.jl`. Essential functionality is still available but distribution-related features may change in the future without a breaking release.
  * `ArrayPartition` (`RecursiveArrayTools.jl` needs to be explicitly imported),
  * `ProjectedPointDistribution` (not exported),
  * `normal_tvector_distribution` (not exported),
  * `projected_distribution` (not exported),
  * `uniform_distribution` (not exported).
* Ability to create non-real `SymplecticStiefel` and `SymplecticGrassmann` manifolds; essential functionality was missing so it was removed until a more developed version is developed.

## [0.9.20] – 2024-06-17

### Added

* implemented parallel transport on the Grassmann manifold with respect to Stiefel representation

### Changed

* since now all exp/log/parallel transport are available for all representations of `Grassmann`,
  these are now also set as defaults, since they are more exact.

## [0.9.19] – 2024-06-12

### Changed

* Updated `Project.toml` compatibility entries.
* Updated CI for Julia 1.11-beta

### Fixed

* a few typos in the doc string of the SPD fixed determinant description
* several other typographical errors throughout the documentation

## [0.9.18] – 2024-05-07

### Added

* added the injectivity radius for the Stiefel manifold with Euclidean metric

## [0.9.17] – 2024-04-23

### Added

* `Hyperrectangle` manifold with boundary.

## [0.9.16] – 2024-04-01

### Changed

* `NonlinearSolve.jl` and `PythonCall.jl` are no longer an upper bounded dependency (bugs were fixed).

## [0.9.15] – 2024-03-24

### Added

* using `DocumenterInterLinks` for links to other Julia packages documentation.
* Implementation of `sectional_curvature`, `sectional_curvature_min` and `sectional_curvature_max` for several manifolds.
* `sectional_curvature_matrix` function and a tutorial on coordinate-free curvature.

### Changed

* `default_vector_transport_method` for `GeneralUnitaryMatrices` other than `Rotations` was changed to `ProjectionTransport`.

### Fixed

* typographical errors in tutorials/working-in-charts.jl.
* several typographical errors in the docs
* unifies to use two backticks ``` `` ``` for math instead of ` $ ` further in the docs

## [0.9.14] – 2024-01-31

### Added

* `rand` on `UnitaryMatrices`
* `rand` on arbitrary `GroupManifold`s and manifolds with `IsGroupManifold` trait
  generating points and elements from the Lie algebra, respectively

## [0.9.13] – 2024-01-24

### Added

* added the real symplectic Grassmann manifold `SymplecticGrassmann`
* Introduce the manifold of `HamiltonianMatrices` and a wrapper for `Hamiltonian` matrices
* introduce `rand(:HamiltonianMatrices)`
* extend `rand` to also `rand!` for `HamiltonianMatrices`, `SymplecticMatrices` and `SymplecticStiefel`
* implement `riemannian_gradient` conversion for `SymplecticMatrices` and `SymplecticGrassmann`
* the new manifold of `MultinomialSymmetricPositiveDefinite` matrices
* `rand!` for `MultinomialDoublyStochastic` and `MultinomialSymmetric`

### Deprecated

* Rename `Symplectic` to `SimplecticMatrices` in order to have a `Symplectic` wrapper for such matrices as well in the future for the next breaking change.
* Rename `SymplecticMatrix` to `SymplecticElement` to clarify that it is the special matrix ``J_{2n}`` and not an arbitrary symplectic matrix.

### Fixed

* a bug that cause `project` for tangent vectors to return wrong results on `MultinomialDoublyStochastic`

## [0.9.12] – 2024-01-21

### Fixed

* Fixed `var` on `TranslationGroup`.

## [0.9.11] – 2023-12-27

### Fixed

* Fixed mixed array index number in-place `parallel_transport_to!` on zero-index `Euclidean`, on Julia 1.6.

## [0.9.10] – 2023-12-27

### Added

* Compatibility with `RecursiveArrayTools` v3.

### Fixed

* Fixed mixed array index number in-place `parallel_transport_to!` on real `Circle`, on Julia 1.6.
* Violations of MD004 lint rule in this file.

## [0.9.9] – 2023-12-25

### Fixed

* introduced a nonzero `atol` for all point and vector checks that compre to zero.
  This makes those checks a bit more relaxed by default and resolves [#630](https://github.com/JuliaManifolds/Manifolds.jl/issues/630).
* `default_estimation_method(M, f)` is deprecated, use `default_approximation_method(M, f)` for your specific method `f` on the manifold `M`.
* `AbstractEstimationMethod` is deprecated, use `AbstractApproximationMethod` instead.

## [0.9.8] - 2023-11-17

### Fixed

* Improved distribution of random vector generation for rotation matrices and complex circle.

## [0.9.7] – 2023-11-14

### Fixed

* Fixed `is_flat` for `CholeskySpace` and `SymmetricPositiveDefinite` with `LogCholeskyMetric` [https://github.com/JuliaManifolds/Manifolds.jl/issues/684](https://github.com/JuliaManifolds/Manifolds.jl/issues/684).

## [0.9.6] - 2023-11-09

### Fixed

* Fixed real coefficient basis for complex circle (an issue exposed by [https://github.com/JuliaManifolds/ManifoldsBase.jl/pull/173](https://github.com/JuliaManifolds/ManifoldsBase.jl/pull/173)).
* Fixed `VeeOrthogonalBasis` test for non-real manifolds.

## [0.9.5] - 2023-11-08

### Changed

* `identity_element` now returns a complex matrix for unitary group.
* `number_of_coordinates` is now exported.

## [0.9.4] - 2023-11-06

### Added

* Functions `inv_diff`, `inv_diff!`, `adjoint_inv_diff` and `adjoint_inv_diff!` that correspond to differentials and pullbacks of group inversion.
* Julia 1.10-rc CI workflow.

### Changed

* Documentation project files are marked as compatible with `BoundaryValueDiffEq` v5.

### Fixed

* Fixed issue with incorrect implementation of `apply_diff_group` in `GroupOperationAction` with left backward and right forward action [#669](https://github.com/JuliaManifolds/Manifolds.jl/issues/669).

## [0.9.3] - 2023-10-28

### Added

* Support for `BoundaryValueDiffEq` v5.

## [0.9.2] - 2023-10-27

### Added

* `rand(G; vector_at=Identity(G))` now works for translation, special orthogonal and special Euclidean groups `G` (issue [#665](https://github.com/JuliaManifolds/Manifolds.jl/issues/665)).
* `get_embedding` now works for `GeneralUnitaryMultiplicationGroup`.
* Github action that checks for NEWS.md changes.

## [0.9.1] - 2023-10-25

### Added

* a new retraction and its inverse for the fixed Rank Manifolds, the orthographic retraction.

## [0.9.0] - 2023-10-24

### Added

* Vector bundles are generalized to fiber bundles. Old `BundleFibers` functionality was reworked to better match mathematical abstractions. Fiber bundle functionality is experimental and minor changes may happen without a breaking release, with the exception of `TangentBundle` which is considered to be stable.
* `RotationTranslationAction` is introduced.

### Changed

* Sizes of all manifolds can now be either encoded in type or stored in a field to avoid over-specialization.
  The default is set to store the size in type parameter (except for `PowerManifold` and its variants), replicating the previous behavior.
  For field storage, pass the `parameter=:field` keyword argument to manifold constructor.
  For example statically sized `CenteredMatrices{m,n}` is now `CenteredMatrices{TypeParameter{Tuple{m,n}}}`, whereas the type of special Euclidean group with field-stored size is `CenteredMatrices{Tuple{Int,Int}}`. Similar change applies to:
  * `CenteredMatrices{m,n}`,
  * `CholeskySpace{N}`,
  * `Elliptope{N,K}`,
  * `Euclidean`,
  * `FixedRankMatrices{m,n,k}`,
  * `KendallsPreShapeSpace{n,k}`,
  * `KendallsShapeSpace{n,k}`,
  * `GeneralLinear{n}`,
  * `GeneralUnitaryMultiplicationGroup{n}`,
  * `GeneralizedGrassmann{n,k}`,
  * `GeneralizedStiefel{n,k}`,
  * `Grassmann{n,k}`,
  * `Heisenberg{n}`,
  * `Hyperbolic{n}`,
  * `MultinomialMatrices{N,M}`,
  * `MultinomialDoublyStochastic{n}`,
  * `MultinomialSymmetric{n}`,
  * `Orthogonal{n}`,
  * `PowerManifold`,
  * `PositiveArrays`,
  * `PositiveMatrices`,
  * `PositiveNumbers`,
  * `ProbabilitySimplex{n}`,
  * `SPDFixedDeterminant{n}`,
  * `SpecialLinear{n}`,
  * `SpecialOrthogonal{n}`,
  * `SpecialUnitary{n}`,
  * `SpecialEuclidean{n}`,
  * `SpecialEuclideanManifold{n}`,
  * `Spectrahedron{n,k}`,
  * `SphereSymmetricMatrices{N}`,
  * `Stiefel{n,k}`,
  * `SymmetricMatrices{N}`,
  * `SymmetricPositiveDefinite{n}`,
  * `SymmetricPositiveSemidefiniteFixedRank{n,k}`,
  * `Symplectic{n}`,
  * `SymplecticStiefel{n,k}`,
  * `TranslationGroup`,
  * `Tucker`.

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

* Argument order for type aliases `RotationActionOnVector` and `RotationTranslationActionOnVector`: most often dispatched on argument is now first.
* A more consistent handling of action direction was introduced. 4-valued `ActionDirection` was split into 2-valued `ActionDirection` (either left or right action) and `GroupActionSide` (action acting from the left or right side). See [https://github.com/JuliaManifolds/Manifolds.jl/issues/637](https://github.com/JuliaManifolds/Manifolds.jl/issues/637) for a design discussion.

### Removed

* `ProductRepr` is removed; please use `ArrayPartition` instead.
* Default methods throwing "not implemented" `ErrorException` for some group-related operations. Standard `MethodError` is now thrown instead.
* `LinearAffineMetric` was deprecated in a previous release and the symbol is now removed.
  Please use `AffineInvariantMetric` instead.
