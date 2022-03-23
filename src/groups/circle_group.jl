@doc raw"""
    CircleGroup <: GroupManifold{Circle{ℂ},MultiplicationOperation}

The circle group is the complex circle ([`Circle(ℂ)`](@ref)) equipped with
the group operation of complex multiplication ([`MultiplicationOperation`](@ref)).
"""
const CircleGroup = GroupManifold{ℂ,Circle{ℂ},MultiplicationOperation}

CircleGroup() = GroupManifold(Circle{ℂ}(), MultiplicationOperation())

@inline function active_traits(f, M::CircleGroup, args...)
    return merge_traits(
        IsGroupManifold(M.op),
        IsDefaultMetric(EuclideanMetric()),
        HasBiinvariantMetric(),
        active_traits(f, M.manifold, args...),
        IsExplicitDecorator(),
    )
end

Base.show(io::IO, ::CircleGroup) = print(io, "CircleGroup()")

adjoint_action(::CircleGroup, p, X) = X

adjoint_action!(::CircleGroup, Y, p, X) = copyto!(Y, X)

identity_element(G::CircleGroup) = 1.0
identity_element(::CircleGroup, p::Number) = one(p)

Base.inv(G::CircleGroup, p::AbstractVector) = map(inv, repeated(G), p)

function inverse_translate(
    ::CircleGroup,
    p::AbstractVector,
    q::AbstractVector,
    ::LeftAction,
)
    return map(/, q, p)
end
function inverse_translate(
    ::CircleGroup,
    p::AbstractVector,
    q::AbstractVector,
    ::RightAction,
)
    return map(/, q, p)
end

lie_bracket(::CircleGroup, X, Y) = zero(X)

lie_bracket!(::CircleGroup, Z, X, Y) = fill!(Z, 0)

translate_diff(::GT, p, q, X, ::ActionDirection) where {GT<:CircleGroup} = map(*, p, X)
function translate_diff(
    ::CircleGroup,
    ::Identity{MultiplicationOperation},
    q,
    X,
    ::ActionDirection,
)
    return X
end

function translate_diff!(G::CircleGroup, Y, p, q, X, conv::ActionDirection)
    return copyto!(Y, translate_diff(G, p, q, X, conv))
end

function exp_lie(::CircleGroup, X)
    return map(X) do imθ
        θ = imag(imθ)
        sinθ, cosθ = sincos(θ)
        return Complex(cosθ, sinθ)
    end
end

exp_lie!(G::CircleGroup, q, X) = (q .= exp_lie(G, X))

function _log_lie(::CircleGroup, q)
    return map(q) do z
        cosθ, sinθ = reim(z)
        θ = atan(sinθ, cosθ)
        return θ * im
    end
end

_log_lie!(G::CircleGroup, X, q) = (X .= _log_lie(G, q))

function number_of_coordinates(G::CircleGroup, B::AbstractBasis)
    return number_of_coordinates(base_manifold(G), B)
end

@doc raw"""
    RealCircleGroup <: GroupManifold{Circle{ℝ},AdditionOperation}

The real circle group is the real circle ([`Circle(ℝ)`](@ref)) equipped with
the group operation of addition ([`AdditionOperation`](@ref)).
"""
const RealCircleGroup = GroupManifold{ℝ,Circle{ℝ},AdditionOperation}

RealCircleGroup() = GroupManifold(Circle{ℝ}(), AdditionOperation())

@inline function active_traits(f, M::RealCircleGroup, args...)
    return merge_traits(
        IsGroupManifold(M.op),
        IsDefaultMetric(EuclideanMetric()),
        HasBiinvariantMetric(),
        active_traits(f, M.manifold, args...),
        IsExplicitDecorator(),
    )
end

Base.show(io::IO, ::RealCircleGroup) = print(io, "RealCircleGroup()")

invariant_metric_dispatch(::RealCircleGroup, ::ActionDirection) = Val(true)

is_default_metric(::RealCircleGroup, ::EuclideanMetric) = true

# Lazy overwrite since this is a rare case of nonmutating foo.
compose(::RealCircleGroup, p, q) = sym_rem(p + q)
compose(::RealCircleGroup, ::Identity{<:AdditionOperation}, q) = sym_rem(q)
compose(::RealCircleGroup, p, ::Identity{<:AdditionOperation}) = sym_rem(p)
function compose(
    ::RealCircleGroup,
    e::Identity{<:AdditionOperation},
    ::Identity{<:AdditionOperation},
)
    return e
end

function compose!(::RealCircleGroup, x, p, q)
    x = sym_rem.(p + q)
    return x
end

identity_element(G::RealCircleGroup) = 0.0
identity_element(::RealCircleGroup, p::AbstractArray) = map(i -> zero(eltype(p)), p)

Base.inv(G::RealCircleGroup, p::AbstractVector) = map(inv, repeated(G), p)

function inverse_translate(
    ::RealCircleGroup,
    p::AbstractVector,
    q::AbstractVector,
    ::LeftAction,
)
    return map((x, y) -> sym_rem(x - y), q, p)
end
function inverse_translate(
    ::RealCircleGroup,
    p::AbstractVector,
    q::AbstractVector,
    ::RightAction,
)
    return map((x, y) -> sym_rem(x - y), q, p)
end

function exp_lie(::RealCircleGroup, X)
    return sym_rem(X)
end

exp_lie!(G::RealCircleGroup, q, X) = (q .= sym_rem(X))
