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

embed(::SymplecticGrassmann, p) = p
embed(::SymplecticGrassmann, p, X) = X
embed!(::SymplecticGrassmann, q, p) = copyto!(q, p)
embed!(::SymplecticGrassmann, Y, p, X) = copyto!(Y, X)
embed!(::SymplecticGrassmann, q, p::StiefelPoint) = copyto!(q, p.value)
embed!(::SymplecticGrassmann, Y, p::StiefelPoint, X::StiefelTVector) = copyto!(Y, X.value)
embed(::SymplecticGrassmann, p::StiefelPoint) = p.value
embed(::SymplecticGrassmann, p::StiefelPoint, X::StiefelTVector) = X.value

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

function get_embedding(::SymplecticGrassmann{TypeParameter{Tuple{n,k}},𝔽}) where {n,k,𝔽}
    return SymplecticStiefel(2n, 2k, 𝔽)
end
function get_embedding(M::SymplecticGrassmann{Tuple{Int,Int},𝔽}) where {𝔽}
    n, k = get_parameter(M.size)
    return SymplecticStiefel(2n, 2k, 𝔽; parameter=:field)
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

@doc raw"""
    riemannian_gradient(M::SymplecticGrassmann, p, Y)

Given a gradient ``Y = \operatorname{grad} \tilde f(p)`` in the embedding ``ℝ^{2n×2k}`` or at
least around the [`SymplecticGrassmann`](@ref) `M` where `p` (the embedding of) a point on `M`,
and the restriction ``\tilde f`` to the [`SymplecticStiefel`] be invariant for the equivalence classes.
In other words ``f(p) = f(qp)`` for ``q \in \mathrm{Sp}(2k, ℝ)``.
Then the Riemannian gradient ``X = \operatorname{grad} f(p)`` is given by

```math
  X = J_{2n}^THJ_{2k}p^{\mathrm{T}}p - J_{2n}^TpJ_{2k}H^{\mathrm{T}}p,
```

where ``J_{2n}`` denotes the [`SymplecticElement`](@ref), and
``H = (I_{2n} - pp^+)J_{2n}^{\mathrm{T}}YJ``.
"""
function riemannian_gradient(::SymplecticGrassmann, p, Y; kwargs...)
    n, k = get_parameter(M.size)
    J = SymplecticElement(p, X)
    # Since J' = -J We can write (J'YJ) = -J * (YJ)
    JTYJ = (-J * (Y * J))
    H = (I - symplectic_inverse_times(SymplecticStiefel(2 * n, 2 * k), p, p)) * JTYJ
    return (-J * (H * J)) * (p' * p) .- JTYJ * (H' * p)
end

function riemannian_gradient!(M::SymplecticGrassmann, X, p, Y; kwargs...)
    n, k = get_parameter(M.size)
    J = SymplecticElement(p, X)
    # Since J' = -J We can write (J'YJ) = -J * (YJ)
    JTYJ = (-J * (Y * J))
    H = (I - symplectic_inverse_times(SymplecticStiefel(2 * n, 2 * k), p, p)) * JTYJ
    X .= (-J * (H * J)) * (p' * p) .- JTYJ * (H' * p)
    return X
end
