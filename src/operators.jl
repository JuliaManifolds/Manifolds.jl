
"""
    AbstractTensorField{𝔽1,TM1<:Manifold{𝔽1},VSIn<:VectorSpaceType,𝔽2,TM2<:Manifold{𝔽2},VSOut<:VectorSpaceType}

An abstract map from vector-valued field over a vector space of type `VSIn` over manifold of
type `TM1` to a vector field over a vector space of type `VSOut` over amnifold of type `TM2`.
"""
abstract type AbstractTensorField{𝔽1,TM1<:Manifold{𝔽1},VSIn<:VectorSpaceType,𝔽2,TM2<:Manifold{𝔽2},VSOut<:VectorSpaceType} end

"""
    apply_operator(op::AbstractTensorField, p, v...)

Apply operator `op` at point `p` to arguments (vectors) `v...`.
"""
function apply_operator(op::AbstractTensorField, p, v...) end

const AbstractScalarValuedField{𝔽,TM,VSIn} = AbstractTensorField{𝔽,TM,VSIn,𝔽,TM,ScalarSpaceType} where {𝔽,TM<:Manifold{𝔽},VSIn<:VectorSpaceType}

"""
    MetricField(M::Manifold)

Multilinear scalar field corresponding to the metric of given manifold `M`.
"""
struct MetricField{𝔽,TM<:Manifold{𝔽}} <: AbstractScalarValuedField{𝔽,TM,TensorProductType{Tuple{TangentSpaceType,TangentSpaceType}}}
    manifold::TM
end

apply_operator(op::MetricField, p, X1, X2) = inner(op.manifold, p, X1, X2)

"""
    AbstractCotangentVectorField{𝔽,TM<:Manifold{𝔽}} <: AbstractScalarValuedField{𝔽,TM,TangentSpaceType}

Defines unique representation of cotangent vectors at each point on a manifold.
"""
abstract type AbstractCotangentVectorField{𝔽,TM<:Manifold{𝔽}} <: AbstractScalarValuedField{𝔽,TM,TangentSpaceType} end

"""
    RieszRepresenterCotangentVectorField(M::Manifold)

Defines Riesz representer representation of cotangent vectors.
"""
struct RieszRepresenterCotangentVectorField{𝔽,TM<:Manifold{𝔽}} <: AbstractCotangentVectorField{𝔽,TM}
    manifold::TM
end

apply_operator(op::RieszRepresenterCotangentVectorField, p, X1, X2) = inner(op.manifold, p, X1, X2)

"""
    get_cotangent_operator(M::Manifold)

Get the default representation of cotangent vectors for manifold `M`. Defaults to
[`RieszRepresenterCotangentVectorField`](@ref).
"""
function get_cotangent_operator(M::Manifold)
    return RieszRepresenterCotangentVectorField(M)
end

"""
    PushforwardField(manifold_in::Manifold, manifold_out::Manifold, f)

Pushforward of function `f` taking arguments in `manifold_in` and values in `manifold_out`.
"""
struct PushforwardField{𝔽1,TM1<:Manifold{𝔽1},𝔽2,TM2<:Manifold{𝔽2},TF} <: AbstractTensorField{𝔽1,TM1,TangentSpaceType,𝔽2,TM2,TangentSpaceType}
    manifold_in::TM1
    manifold_out::TM2
    f::TF
end

# TODO: get_coordinates for `PushforwardField`

function apply_operator(op::PushforwardField, p, X)
    # TODO
end

