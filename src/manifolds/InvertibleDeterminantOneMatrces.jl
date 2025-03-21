#
#
#
# This is both a new manifold and a small experiment on how to maybe replace
# the complicated traits system in ManifoldsBase.
# (because it is a bit annoyingly implicit)

"""
    Specify that a manifold has a certain property
"""
abstract type ManifoldProperty end

struct InvertibleMatrix <: ManifoldProperty end
struct MatrixDeterminant{S} <: ManifoldProperty end
# ideas for further properties
struct IsometricEmbedding <: ManifoldProperty end
struct DefaultMetric{G<:AbstractMetric} <: ManifoldProperty end

"""
    MatrixManifold{𝔽, Type, Properties} <: AbstractManifold{𝔽}

A generic type for manifolds that are represented by matrices ``p ∈ 𝔽^{n×m}`` or arrays in general

* `𝔽` represents the field the manifold is defined over using for example [`AbstractNumbers`](@refref `ManifoldsBase.AbstractNumbers`)
* `Type` represents the [`TypeParameter`(@extref `ManifoldsBase.TypeParameter`) to allow to store the size information in the type to dispatch on
* `Properties` allows to specify for example a [`ManifoldProperty`](@ref) or a tuple of them
  in order to unify implementations. the order is important and should be from most specific (first) to least
"""
struct MatrixManifold{𝔽,Type,Properties} <: AbstractManifold{𝔽}
    size::Type
end

const InvertibleMatricesDeterminantOne{𝔽,T} =
    MatrixManifold{𝔽,T,Tuple{Manifolds.InvertibleMatrix,MatrixDeterminant{1}}}

function InvertibleMatricesDeterminantOne(
    n::Int,
    field::AbstractNumbers=ℝ;
    parameter::Symbol=:type,
)
    size = wrap_type_parameter(parameter, (n,))
    return InvertibleMatricesDeterminantOne{field,typeof(size)}(size)
end

#
#
# Proof of concept: check
function check_point(
    M::MatrixManifold{𝔽,T,Pr},
    p;
    kwargs...,
) where {𝔽,T,n,Pr<:Tuple{MatrixDeterminant{n},Vararg{<:Manifolds.ManifoldProperty}}}
    if det(p) != n
        return DomainError(
            det(p),
            "The point $(p) does not lie on $(M), since its determinant is $(det(p)) and not $(n).",
        )
    end
    return check_point(
        MatrixManifold{𝔽,T,Tuple{Pr.parameters[2:end]...}}(M.size),
        p;
        kwargs...,
    )
end
function check_point(
    M::MatrixManifold{𝔽,T,Pr},
    p;
    kwargs...,
) where {𝔽,T,Pr<:Tuple{InvertibleMatrix,Vararg{<:Manifolds.ManifoldProperty}}}
    if det(p) == 0
        return DomainError(
            det(p),
            "The point $(p) does not lie on $(M), since its determinant is zero and hence it is not invertible.",
        )
    end
    return check_point(
        MatrixManifold{𝔽,T,Tuple{Pr.parameters[2:end]...}}(M.size),
        p;
        kwargs...,
    )
end
# generics, pass through and end
function check_point(
    M::MatrixManifold{𝔽,T,Pr},
    p;
    kwargs...,
) where {𝔽,T,Pr<:Tuple{<:ManifoldProperty,Vararg{<:ManifoldProperty}}}
    return check_point(MatrixManifold{𝔽,T,Pr.parameters[2:end]...}, p; kwargs...)
end
check_point(M::MatrixManifold{𝔽,T,Tuple{}}, p; kwargs...) where {𝔽,T} = nothing
