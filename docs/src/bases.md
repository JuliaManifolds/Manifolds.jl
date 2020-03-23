# Bases

The following functions and types provide support for bases of the tangent space of different manifolds.
An orthonormal basis of the tangent space $T_p \mathcal M$ of (real) dimension $n$ has a real-coefficient basis $e_1, e_2, …, e_n$ if $\mathrm{Re}(g_p(e_i, e_j)) = δ_{ij}$ for each $i,j ∈ \{1, 2, …, n\}$ where $g_p$ is the Riemannian metric at point $p$.
A vector $X$ from the tangent space $T_p \mathcal M$ can be expressed in Einstein notation as a sum $X = X^i e_i$, where (real) coefficients $X^i$ are calculated as $X^i = \mathrm{Re}(g_p(X, e_i))$.

The main types are:
* [`DefaultOrthonormalBasis`](@ref), which is designed to work when no special properties of the tangent space basis are required.
   It is designed to make [`get_coordinates`](@ref) and [`get_vector`](@ref) fast.
* [`DiagonalizingOrthonormalBasis`](@ref), which diagonalizes the curvature tensor and makes the curvature in the selected direction equal to 0.
* [`ProjectedOrthonormalBasis`](@ref), which projects a basis of the ambient space and orthonormalizes projections to obtain a basis in a generic way.
* [`CachedBasis`](@ref), which stores (explicitly or implicitly) a precomputed basis at a certain point.

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
