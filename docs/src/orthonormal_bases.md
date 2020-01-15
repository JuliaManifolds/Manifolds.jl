# Orthonormal bases

Following functions and types provide support for orthonormal bases of the tangent space of different manifolds.

The main types are:
* [`ArbitraryOrthonormalBasis`](@ref) that is designed to work when no special properties of the tangent space basis are required. It is designed to make [`get_coordinates`](@ref) and [`get_vector`](@ref) fast.
* [`DiagonalizingOrthonormalBasis`](@ref) that diagonalizes the curvature tensor and makes the curvature in selected direction equal to 0.
* [`ProjectedOrthonormalBasis`](@ref) that projects a basis of the ambient space and orthonormalizes projections to obtain a basis in a generic way.
* Subtypes of [`AbstractPrecomputedOrthonormalBasis`](@ref) that work by storing (either explicitly or implicitly) a precomputed basis at a certain point. [`ArbitraryOrthonormalBasis`](@ref) doesn't require precomputing but [`DiagonalizingOrthonormalBasis`](@ref) and [`DiagonalizingOrthonormalBasis`](@ref) usually do.

The main functions are:
* [`basis`](@ref) precomputes basis at a certain point.
* [`get_coordinates`](@ref) returns coordinates of a tangent vector.
* [`get_vector`](@ref) returns vector for specified coordinates.
* [`vectors`](@ref) returns a vector of basis vectors (calling it should be avoided for high-dimensional manifolds).

```@autodocs
Modules = [Manifolds]
Pages = ["orthonormal_bases.jl"]
Order = [:type, :function]
```
