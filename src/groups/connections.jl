
"""
    AbstractCartanSchoutenConnection

Abstract type for Cartan-Schouten connections, that is connections whose geodesics
going through group identity are one-parameter subgroups. See [PennecLorenzi:2020](@cite) for details.
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
    exp(M::ConnectionManifold{𝔽,<:AbstractDecoratorManifold{𝔽},<:AbstractCartanSchoutenConnection}, p, X) where {𝔽}

Compute the exponential map on the [`ConnectionManifold`](@ref) `M` with a Cartan-Schouten
connection. See Sections 5.3.2 and 5.3.3 of [PennecLorenzi:2020](@cite) for details.
"""
function exp(
    M::ConnectionManifold{
        𝔽,
        <:AbstractDecoratorManifold{𝔽},
        <:AbstractCartanSchoutenConnection,
    },
    p,
    X,
) where {𝔽}
    Y = inverse_translate_diff(M.manifold, p, p, X, LeftForwardAction())
    return compose(M.manifold, p, exp_lie(M.manifold, Y))
end
function exp(
    M::ConnectionManifold{
        𝔽,
        <:AbstractDecoratorManifold{𝔽},
        <:AbstractCartanSchoutenConnection,
    },
    p,
    X,
    t::Number,
) where {𝔽}
    return exp(M, p, t * X)
end

function exp!(
    M::ConnectionManifold{
        𝔽,
        <:AbstractDecoratorManifold{𝔽},
        <:AbstractCartanSchoutenConnection,
    },
    q,
    p,
    X,
) where {𝔽}
    Y = inverse_translate_diff(M.manifold, p, p, X, LeftForwardAction())
    return compose!(M.manifold, q, p, exp_lie(M.manifold, Y))
end

function exp!(
    M::ConnectionManifold{
        𝔽,
        <:AbstractDecoratorManifold{𝔽},
        <:AbstractCartanSchoutenConnection,
    },
    q,
    p,
    X,
    t::Number,
) where {𝔽}
    return exp!(M, q, p, t * X)
end

"""
    log(M::ConnectionManifold{𝔽,<:AbstractDecoratorManifold{𝔽},<:AbstractCartanSchoutenConnection}, p, q) where {𝔽}

Compute the logarithmic map on the [`ConnectionManifold`](@ref) `M` with a Cartan-Schouten
connection. See Sections 5.3.2 and 5.3.3 of [PennecLorenzi:2020](@cite) for details.
"""
function log(
    M::ConnectionManifold{
        𝔽,
        <:AbstractDecoratorManifold{𝔽},
        <:AbstractCartanSchoutenConnection,
    },
    p,
    q,
) where {𝔽}
    pinvq = compose(M.manifold, inv(M.manifold, p), q)
    Y = log_lie(M.manifold, pinvq)
    return translate_diff(M.manifold, p, Identity(M.manifold), Y, LeftForwardAction())
end

function log!(
    M::ConnectionManifold{
        𝔽,
        <:AbstractDecoratorManifold{𝔽},
        <:AbstractCartanSchoutenConnection,
    },
    Y,
    p,
    q,
) where {𝔽}
    pinvq = compose(M.manifold, inv(M.manifold, p), q)
    log_lie!(M.manifold, Y, pinvq)
    return translate_diff!(M.manifold, Y, p, Identity(M.manifold), Y)
end

"""
    parallel_transport_to(M::CartanSchoutenMinusGroup, p, X, q)

Transport tangent vector `X` at point `p` on the group manifold `M` with the
[`CartanSchoutenMinus`](@ref) connection to point `q`. See [PennecLorenzi:2020](@cite) for details.
"""
function parallel_transport_to(M::CartanSchoutenMinusGroup, p, X, q)
    return inverse_translate_diff(M.manifold, q, p, X, LeftForwardAction())
end

function parallel_transport_to!(M::CartanSchoutenMinusGroup, Y, p, X, q)
    return inverse_translate_diff!(M.manifold, Y, q, p, X, LeftForwardAction())
end

"""
    vector_transport_to(M::CartanSchoutenPlusGroup, p, X, q)

Transport tangent vector `X` at point `p` on the group manifold `M` with the
[`CartanSchoutenPlus`](@ref) connection to point `q`. See [PennecLorenzi:2020](@cite) for details.
"""
parallel_transport_to(M::CartanSchoutenPlusGroup, p, X, q)

function parallel_transport_to!(M::CartanSchoutenPlusGroup, Y, p, X, q)
    return inverse_translate_diff!(M.manifold, Y, q, p, X, RightBackwardAction())
end

"""
    parallel_transport_direction(M::CartanSchoutenZeroGroup, ::Identity, X, d)

Transport tangent vector `X` at identity on the group manifold with the
[`CartanSchoutenZero`](@ref) connection in the direction `d`. See [PennecLorenzi:2020](@cite) for details.
"""
function parallel_transport_direction(M::CartanSchoutenZeroGroup, p::Identity, X, d)
    dexp_half = exp_lie(M.manifold, d / 2)
    Y = translate_diff(M.manifold, dexp_half, p, X, RightBackwardAction())
    return translate_diff(M.manifold, dexp_half, p, Y, LeftForwardAction())
end

function parallel_transport_direction!(M::CartanSchoutenZeroGroup, Y, p::Identity, X, d)
    dexp_half = exp_lie(M.manifold, d / 2)
    translate_diff!(M.manifold, Y, dexp_half, p, X, RightBackwardAction())
    return translate_diff!(M.manifold, Y, dexp_half, p, Y, LeftForwardAction())
end

"""
    parallel_transport_to(M::CartanSchoutenZeroGroup, p::Identity, X, q)

Transport vector `X` at identity of group `M` equipped with the [`CartanSchoutenZero`](@ref)
connection to point `q` using parallel transport.
"""
function parallel_transport_to(M::CartanSchoutenZeroGroup, p::Identity, X, q)
    d = log_lie(M.manifold, q)
    return parallel_transport_direction(M, p, X, d)
end

function parallel_transport_to!(M::CartanSchoutenZeroGroup, Y, p::Identity, X, q)
    d = log_lie(M.manifold, q)
    return parallel_transport_direction!(M, Y, p, X, d)
end
