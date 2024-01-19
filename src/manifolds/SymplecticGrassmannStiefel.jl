@doc raw"""
    check_point(M::SymplecticGrassmann, p; kwargs...)

Check whether `p` is a valid point on the [`SymplecticGrassmann`](@ref),
``\operatorname{SpGr}(2n, 2k)`` manifold by verifying that it
is a valid representer of an equivalence class of the corersponding
[`SymplecticStiefel`](@ref) manifold.
"""
function check_point(M::SymplecticGrassmann, p; kwargs...)
    n, k = get_parameter(M.size)
    return check_point(SymplecticStiefel(2 * n, 2 * k), p; kwargs...)
end

@doc raw"""
    check_vector(M::SymplecticGrassmann, p, X; kwargs...)

Check whether `X` is a valid tangent vector at `p` on the [`SymplecticGrassmann`](@ref),
``\operatorname{SpGr}(2n, 2k)`` manifold by verifying that it
is a valid representer of an equivalence class of the corersponding
[`SymplecticStiefel`](@ref) manifolds tangent space at `p`.
"""
function check_vector(M::SymplecticGrassmann, p, X; kwargs...)
    n, k = get_parameter(M.size)
    return check_vector(SymplecticStiefel(2 * n, 2 * k), p, X; kwargs...)
end

@doc raw"""
    exp(::SymplecticGrassmann, p, X)
    exp!(M::SymplecticGrassmann, q, p, X)

Compute the exponential mapping

```math
  \exp\colon T\mathrm{SpGr}(2n, 2k) → \mathrm{SpGr}(2n, 2k)
```

when representing points and tangent vectors as symplectic bases and their tangents, i.e.
on the [`SymplecticStiefel`](@ref) manifold. Then we can just pass this on to [`exp(::SymplecticStiefel, p, X)`](@ref).
"""
exp(::SymplecticGrassmann, p, X)

function exp!(M::SymplecticGrassmann, q, p, X)
    n, k = get_parameter(M.size)
    exp!(SymplecticStiefel(2 * n, 2 * k), q, p, X)
    return q
end

@doc raw"""
    inverse_retract(::SymplecticGrassmann, p, q, ::CayleyInverseRetraction)
    inverse_retract!(::SymplecticGrassmann, q, p, X, ::CayleyInverseRetraction)

Compute the Cayley Inverse Retraction on the Symplectic Grassmann manifold,
when the points are represented as symplectic bases, i.e. on the [`SymplecticStiefel`](@ref).

Here we can directly employ the `CaleyInverseRetraction` on the symplectic Stiefel manifold.
"""
inverse_retract(::SymplecticGrassmann, p, q, ::CayleyInverseRetraction)

function inverse_retract_cayley!(M::SymplecticGrassmann, X, p, q)
    n, k = get_parameter(M.size)
    return inverse_retract_cayley!(SymplecticStiefel(2 * n, 2 * k), X, p, q)
end

@doc raw"""
    retract(::SymplecticGrassmann, p, X, ::CayleyRetraction)
    retract!(::SymplecticGrassmann, q, p, X, ::CayleyRetraction)

Compute the Cayley retraction on the Symplectic Grassmann manifold,
when the points are represented as symplectic bases, i.e. on the [`SymplecticStiefel`](@ref).

Here we can directly employ the `CaleyRetraction` on the symplectic Stiefel manifold.
"""
retract(::SymplecticGrassmann, p, X, ::CayleyRetraction)

function retract_cayley!(M::SymplecticGrassmann, q, p, X, t::Number)
    n, k = get_parameter(M.size)
    retract_cayley!(SymplecticStiefel(2 * n, 2 * k), q, p, X, t)
    return q
end
