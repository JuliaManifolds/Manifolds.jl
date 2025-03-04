@doc raw"""
    CircleGroup <: GroupManifold{Circle{ℂ},MultiplicationOperation}

The circle group is the complex circle ([`Circle(ℂ)`](@ref)) equipped with
the group operation of complex multiplication ([`MultiplicationOperation`](@ref)).
"""
const CircleGroup =
    GroupManifold{ℂ,Circle{ℂ},MultiplicationOperation,TangentVectorRepresentation}

function CircleGroup()
    _lie_groups_depwarn_move(CircleGroup)
    return GroupManifold(
        Circle{ℂ}(),
        MultiplicationOperation(),
        TangentVectorRepresentation(),
    )
end

@inline function active_traits(f, M::CircleGroup, args...)
    if is_metric_function(f)
        #pass to Euclidean by default - but keep Group Decorator for the retraction
        return merge_traits(
            IsGroupManifold(M.op, TangentVectorRepresentation()),
            IsExplicitDecorator(),
        )
    else
        return merge_traits(
            IsGroupManifold(M.op, TangentVectorRepresentation()),
            IsDefaultMetric(EuclideanMetric()),
            active_traits(f, M.manifold, args...),
            IsExplicitDecorator(), #pass to Euclidean by default/last fallback
        )
    end
end

Base.show(io::IO, ::CircleGroup) = print(io, "CircleGroup()")

adjoint_action(::CircleGroup, p, X, ::LeftAction) = X
adjoint_action(::CircleGroup, ::Identity, X, ::LeftAction) = X
adjoint_action(::CircleGroup, p, X, ::RightAction) = X
adjoint_action(::CircleGroup, ::Identity, X, ::RightAction) = X

adjoint_action!(::CircleGroup, Y, p, X, ::LeftAction) = copyto!(Y, X)
adjoint_action!(::CircleGroup, Y, p, X, ::RightAction) = copyto!(Y, X)

function compose(
    ::MultiplicationGroupTrait,
    G::CircleGroup,
    p::AbstractArray{<:Any,0},
    q::AbstractArray{<:Any,0},
)
    return map((pp, qq) -> compose(G, pp, qq), p, q)
end

function compose!(
    ::MultiplicationGroupTrait,
    G::CircleGroup,
    x,
    p::AbstractArray{<:Any,0},
    q::AbstractArray{<:Any,0},
)
    return copyto!(x, compose(G, p, q))
end

has_biinvariant_metric(::CircleGroup) = true

has_invariant_metric(::CircleGroup, ::ActionDirectionAndSide) = true

identity_element(G::CircleGroup) = 1.0
identity_element(::CircleGroup, p::Number) = one(p)

Base.inv(G::CircleGroup, p::AbstractArray{<:Any,0}) = map(pp -> inv(G, pp), p)

function inverse_translate(
    ::CircleGroup,
    p::AbstractArray{<:Any,0},
    q::AbstractArray{<:Any,0},
    ::LeftForwardAction,
)
    return map(/, q, p)
end
function inverse_translate(
    ::CircleGroup,
    p::AbstractArray{<:Any,0},
    q::AbstractArray{<:Any,0},
    ::RightBackwardAction,
)
    return map(/, q, p)
end

lie_bracket(::CircleGroup, X, Y) = zero(X)

lie_bracket!(::CircleGroup, Z, X, Y) = fill!(Z, 0)

translate_diff(::CircleGroup, p, q, X) = map(*, p, X)
translate_diff(::CircleGroup, p::Identity{MultiplicationOperation}, q, X) = X
for AD in [LeftForwardAction, RightForwardAction, LeftBackwardAction, RightBackwardAction]
    @eval begin
        function translate_diff(G::CircleGroup, p, q, X, ::$AD)
            return translate_diff(G, p, q, X)
        end
        function translate_diff(
            ::CircleGroup,
            ::Identity{MultiplicationOperation},
            q,
            X,
            ::$AD,
        )
            return X
        end
    end
end

_common_translate_diff!(G, Y, p, q, X, conv) = copyto!(Y, translate_diff(G, p, q, X, conv))
function translate_diff!(G::CircleGroup, Y, p, q, X, conv::LeftForwardAction)
    return _common_translate_diff!(G, Y, p, q, X, conv)
end
function translate_diff!(G::CircleGroup, Y, p, q, X, conv::RightForwardAction)
    return _common_translate_diff!(G, Y, p, q, X, conv)
end
function translate_diff!(G::CircleGroup, Y, p, q, X, conv::LeftBackwardAction)
    return _common_translate_diff!(G, Y, p, q, X, conv)
end
function translate_diff!(G::CircleGroup, Y, p, q, X, conv::RightBackwardAction)
    return _common_translate_diff!(G, Y, p, q, X, conv)
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
const RealCircleGroup =
    GroupManifold{ℝ,Circle{ℝ},AdditionOperation,LeftInvariantRepresentation}

function RealCircleGroup()
    return GroupManifold(Circle{ℝ}(), AdditionOperation(), LeftInvariantRepresentation())
end

@inline function active_traits(f, M::RealCircleGroup, args...)
    if is_metric_function(f)
        #pass to Euclidean by default - but keep Group Decorator for the retraction
        return merge_traits(IsGroupManifold(M.op, M.vectors), IsExplicitDecorator())
    else
        return merge_traits(
            IsGroupManifold(M.op, M.vectors),
            HasBiinvariantMetric(),
            IsDefaultMetric(EuclideanMetric()),
            active_traits(f, M.manifold, args...),
            IsExplicitDecorator(), #pass to Euclidean by default/last fallback
        )
    end
end

adjoint_action(::RealCircleGroup, p, X, ::LeftAction) = X
adjoint_action(::RealCircleGroup, p, X, ::RightAction) = X
adjoint_action(::RealCircleGroup, ::Identity, X, ::LeftAction) = X
adjoint_action(::RealCircleGroup, ::Identity, X, ::RightAction) = X

for AD in [LeftAction, RightAction]
    @eval begin
        adjoint_action!(::RealCircleGroup, Y, p, X, ::$AD) = copyto!(Y, X)
    end
end

Base.show(io::IO, ::RealCircleGroup) = print(io, "RealCircleGroup()")

is_default_metric(::RealCircleGroup, ::EuclideanMetric) = true

# Lazy overwrite since this is a rare case of nonmutating foo.
compose(::RealCircleGroup, p, q) = sym_rem(p + q)
compose(::RealCircleGroup, ::Identity{AdditionOperation}, q) = sym_rem(q)
compose(::RealCircleGroup, p, ::Identity{AdditionOperation}) = sym_rem(p)
function compose(
    ::RealCircleGroup,
    e::Identity{AdditionOperation},
    ::Identity{AdditionOperation},
)
    return e
end

compose!(::RealCircleGroup, x, p, q) = copyto!(x, sym_rem(p + q))
compose!(::RealCircleGroup, x, ::Identity{AdditionOperation}, q) = copyto!(x, sym_rem(q))
compose!(::RealCircleGroup, x, p, ::Identity{AdditionOperation}) = copyto!(x, sym_rem(p))
function compose!(
    ::RealCircleGroup,
    ::Identity{AdditionOperation},
    e::Identity{AdditionOperation},
    ::Identity{AdditionOperation},
)
    return e
end

identity_element(G::RealCircleGroup) = 0.0
identity_element(::RealCircleGroup, p) = zero(p)

Base.inv(G::RealCircleGroup, p::AbstractArray{<:Any,0}) = map(pp -> inv(G, pp), p)

function inverse_translate(
    ::RealCircleGroup,
    p::AbstractArray{<:Any,0},
    q::AbstractArray{<:Any,0},
    ::LeftForwardAction,
)
    return map((x, y) -> sym_rem(x - y), q, p)
end
function inverse_translate(
    ::RealCircleGroup,
    p::AbstractArray{<:Any,0},
    q::AbstractArray{<:Any,0},
    ::RightBackwardAction,
)
    return map((x, y) -> sym_rem(x - y), q, p)
end

function exp_lie(::RealCircleGroup, X)
    return sym_rem(X)
end

exp_lie!(::RealCircleGroup, q, X) = (q .= sym_rem(X))

translate_diff(::RealCircleGroup, p, q, X) = X
for AD in [LeftForwardAction, RightForwardAction, LeftBackwardAction, RightBackwardAction]
    @eval begin
        function translate_diff(::RealCircleGroup, p, q, X, ::$AD)
            return X
        end
        function translate_diff(
            ::RealCircleGroup,
            ::Identity{AdditionOperation},
            q,
            X,
            ::$AD,
        )
            return X
        end
    end
end
