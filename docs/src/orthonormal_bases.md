# Orthonormal bases

The following functions and types provide support for orthonormal bases of the tangent space of different manifolds.
An orthonormal basis of the tangent space $T_x ℳ$ of (real) dimension $N$ has a real-coefficient basis $e_1, e_2, \dots, e_N$ if $\mathrm{Re}(g_x(e_i, e_j)) = \delta_{ij}$ for each $i,j \in \{1, 2, \dots, N\}$ where $g_x$ is the Riemannian metric at point $x$.
A vector $v$ from the tangent space $T_x ℳ$ can be expressed as a sum $v = v^i e_i$ where coefficients $v^i$ are calculated as $v^i = \mathrm{Re}(g_x(v, e_i))$.

The main types are:
* [`ArbitraryOrthonormalBasis`](@ref), which is designed to work when no special properties of the tangent space basis are required.
   It is designed to make [`get_coordinates`](@ref) and [`get_vector`](@ref) fast.
* [`DiagonalizingOrthonormalBasis`](@ref), which diagonalizes the curvature tensor and makes the curvature in the selected direction equal to 0.
* [`ProjectedOrthonormalBasis`](@ref), which projects a basis of the ambient space and orthonormalizes projections to obtain a basis in a generic way.
* Subtypes of [`AbstractPrecomputedOrthonormalBasis`](@ref), which work by storing (either explicitly or implicitly) a precomputed basis at a certain point.
  [`ArbitraryOrthonormalBasis`](@ref) doesn't require precomputing but [`DiagonalizingOrthonormalBasis`](@ref) and [`DiagonalizingOrthonormalBasis`](@ref) usually do.

The main functions are:
* [`get_basis`](@ref) precomputes a basis at a certain point.
* [`get_coordinates`](@ref) returns coordinates of a tangent vector.
* [`get_vector`](@ref) returns a vector for the specified coordinates.
* [`get_vectors`](@ref) returns a vector of basis vectors (calling it should be avoided for high-dimensional manifolds).

```@autodocs
Modules = [Manifolds]
Pages = ["orthonormal_bases.jl"]
Order = [:type, :function]
```
