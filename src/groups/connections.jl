
"""
    AbstractCartanSchoutenConnection

Abstract type for Cartan-Schouten connections, that is connections whose geodesics
going through group identity are one-parameter subgroups. See [^Pennec2020] for details.

[^Pennec2020]:
    > X. Pennec and M. Lorenzi, “5 - Beyond Riemannian geometry: The affine connection
    > setting for transformation groups,” in Riemannian Geometric Statistics in Medical Image
    > Analysis, X. Pennec, S. Sommer, and T. Fletcher, Eds. Academic Press, 2020, pp. 169–229.
    > doi: 10.1016/B978-0-12-814725-2.00012-1.
"""
abstract type AbstractCartanSchoutenConnection <: AbstractAffineConnection end

"""
    CartanSchoutenMinus

The unique Cartan-Schouten connection such that all left-invariant vector fields are
globally defined by their value at identity. It is biinvariant with respect to the group
operation.
"""
struct CartanSchoutenMinus <: AbstractCartanSchoutenConnection end

"""
    CartanSchoutenPlus

The unique Cartan-Schouten connection such that all right-invariant vector fields are
globally defined by their value at identity. It is biinvariant with respect to the group
operation.
"""
struct CartanSchoutenPlus <: AbstractCartanSchoutenConnection end

"""
    CartanSchoutenZero

The unique torsion-free Cartan-Schouten connection. It is biinvariant with respect to the
group operation.

If the metric on the underlying manifold is bi-invariant then it is equivalent to the
Levi-Civita connection of that metric.
"""
struct CartanSchoutenZero <: AbstractCartanSchoutenConnection end

const CartanSchoutenMinusGroup{𝔽,M} = ConnectionManifold{𝔽,M,CartanSchoutenMinus}
const CartanSchoutenPlusGroup{𝔽,M} = ConnectionManifold{𝔽,M,CartanSchoutenPlus}
const CartanSchoutenZeroGroup{𝔽,M} = ConnectionManifold{𝔽,M,CartanSchoutenZero}

"""
    exp!(M::ConnectionManifold{𝔽,<:AbstractGroupManifold{𝔽},<:AbstractCartanSchoutenConnection}, q, p, X) where {𝔽}

Compute the exponential map on the [`ConnectionManifold`](@ref) `M` with a Cartan-Schouten
connection. See Sections 5.3.2 and 5.3.3 of [^Pennec2020] for details.

[^Pennec2020]:
    > X. Pennec and M. Lorenzi, “5 - Beyond Riemannian geometry: The affine connection
    > setting for transformation groups,” in Riemannian Geometric Statistics in Medical Image
    > Analysis, X. Pennec, S. Sommer, and T. Fletcher, Eds. Academic Press, 2020, pp. 169–229.
    > doi: 10.1016/B978-0-12-814725-2.00012-1.
"""
function exp!(
    M::ConnectionManifold{𝔽,<:AbstractGroupManifold{𝔽},<:AbstractCartanSchoutenConnection},
    q,
    p,
    X,
) where {𝔽}
    Y = inverse_translate_diff(M.manifold, q, p, X)
    return compose!(M.manifold, q, p, group_exp(M.manifold, Y))
end

"""
    log!(M::ConnectionManifold{𝔽,<:AbstractGroupManifold{𝔽},<:AbstractCartanSchoutenConnection}, Y, p, q) where {𝔽}

Compute the logarithmic map on the [`ConnectionManifold`](@ref) `M` with a Cartan-Schouten
connection. See Sections 5.3.2 and 5.3.3 of [^Pennec2020] for details.

[^Pennec2020]:
    > X. Pennec and M. Lorenzi, “5 - Beyond Riemannian geometry: The affine connection
    > setting for transformation groups,” in Riemannian Geometric Statistics in Medical Image
    > Analysis, X. Pennec, S. Sommer, and T. Fletcher, Eds. Academic Press, 2020, pp. 169–229.
    > doi: 10.1016/B978-0-12-814725-2.00012-1.
"""
function log!(
    M::ConnectionManifold{𝔽,<:AbstractGroupManifold{𝔽},<:AbstractCartanSchoutenConnection},
    Y,
    p,
    q,
) where {𝔽}
    pinvq = compose(M.manifold, inv(M.manifold, p), q)
    group_log!(M.manifold, Y, pinvq)
    return translate_diff!(M.manifold, Y, p, Identity(M.manifold, p), Y)
end

"""
    vector_transport_to(M::CartanSchoutenMinusGroup, p, X, q, ::ParallelTransport)

Transport tangent vector `X` at point `p` on the group manifold `M` with the
[`CartanSchoutenMinus`](@ref) connection to point `q`. See [^Pennec2020] for details.

[^Pennec2020]:
    > X. Pennec and M. Lorenzi, “5 - Beyond Riemannian geometry: The affine connection
    > setting for transformation groups,” in Riemannian Geometric Statistics in Medical Image
    > Analysis, X. Pennec, S. Sommer, and T. Fletcher, Eds. Academic Press, 2020, pp. 169–229.
    > doi: 10.1016/B978-0-12-814725-2.00012-1.
"""
vector_transport_to(M::CartanSchoutenMinusGroup, p, X, q, ::ParallelTransport)

function vector_transport_to!(M::CartanSchoutenMinusGroup, Y, p, X, q, ::ParallelTransport)
    return inverse_translate_diff!(M.manifold, Y, q, p, X, LeftAction())
end

"""
    vector_transport_to(M::CartanSchoutenPlusGroup, p, X, q, ::ParallelTransport)

Transport tangent vector `X` at point `p` on the group manifold `M` with the
[`CartanSchoutenPlus`](@ref) connection to point `q`. See [^Pennec2020] for details.

[^Pennec2020]:
    > X. Pennec and M. Lorenzi, “5 - Beyond Riemannian geometry: The affine connection
    > setting for transformation groups,” in Riemannian Geometric Statistics in Medical Image
    > Analysis, X. Pennec, S. Sommer, and T. Fletcher, Eds. Academic Press, 2020, pp. 169–229.
    > doi: 10.1016/B978-0-12-814725-2.00012-1.
"""
vector_transport_to(M::CartanSchoutenPlusGroup, p, X, q, ::ParallelTransport)

function vector_transport_to!(M::CartanSchoutenPlusGroup, Y, p, X, q, ::ParallelTransport)
    return inverse_translate_diff!(M.manifold, Y, q, p, X, RightAction())
end

"""
    vector_transport_direction(M::CartanSchoutenZeroGroup, ::Identity, X, d, ::ParallelTransport)

Transport tangent vector `X` at identity on the group manifold with the
[`CartanSchoutenZero`](@ref) connection in the direction `d`. See [^Pennec2020] for details.

[^Pennec2020]:
    > X. Pennec and M. Lorenzi, “5 - Beyond Riemannian geometry: The affine connection
    > setting for transformation groups,” in Riemannian Geometric Statistics in Medical Image
    > Analysis, X. Pennec, S. Sommer, and T. Fletcher, Eds. Academic Press, 2020, pp. 169–229.
    > doi: 10.1016/B978-0-12-814725-2.00012-1.
"""
vector_transport_direction(
    M::CartanSchoutenZeroGroup,
    Y,
    ::Identity,
    X,
    d,
    ::ParallelTransport,
)

function vector_transport_direction!(
    M::CartanSchoutenZeroGroup,
    Y,
    p::Identity,
    X,
    d,
    ::ParallelTransport,
)
    dexp_half = group_exp(M.manifold, d / 2)
    translate_diff!(M.manifold, Y, dexp_half, p, X, RightAction())
    return translate_diff!(M.manifold, Y, dexp_half, p, Y, LeftAction())
end

"""
    vector_transport_to(M::CartanSchoutenZeroGroup, ::Identity, X, q, m::ParallelTransport)

Transport vector `X` at identity of group `M` equipped with the [`CartanSchoutenZero`](@ref)
connection to point `q` using parallel transport.
"""
vector_transport_to(::CartanSchoutenZeroGroup, ::Identity, X, q, ::ParallelTransport)

function vector_transport_to!(
    M::CartanSchoutenZeroGroup,
    Y,
    p::Identity,
    X,
    q,
    m::ParallelTransport,
)
    d = group_log(M.manifold, q)
    return vector_transport_direction!(M, Y, p, X, d, m)
end
