"""
    VectorSpaceType

Abstract type for tangent spaces, cotangent spaces, their tensor products,
exterior products, etc.

Every vector space `fiber` is supposed to provide:
* a method of constructing vectors,
* basic operations: addition, subtraction, multiplication by a scalar
  and negation (unary minus),
* [`zero_vector!(fiber, X, p)`](@ref) to construct zero vectors at point `p`,
* `allocate(X)` and `allocate(X, T)` for vector `X` and type `T`,
* `copyto!(X, Y)` for vectors `X` and `Y`,
* `number_eltype(v)` for vector `v`,
* [`vector_space_dimension(::VectorBundleFibers{<:typeof(fiber)}) where fiber`](@ref).

Optionally:
* inner product via `inner` (used to provide Riemannian metric on vector
  bundles),
* [`flat`](@ref) and [`sharp`](@ref),
* `norm` (by default uses `inner`),
* [`project`](@ref) (for embedded vector spaces),
* [`representation_size`](@ref) (if support for [`ProductArray`](@ref) is desired),
* broadcasting for basic operations.
"""
abstract type VectorSpaceType{𝔽} end


struct TangentSpaceType{𝔽} <: VectorSpaceType{𝔽} end

TangentSpaceType(𝔽) = TangentSpaceType{𝔽}()

struct CotangentSpaceType{𝔽} <: VectorSpaceType{𝔽} end

CotangentSpaceType(𝔽) = CotangentSpaceType{𝔽}()

const TCoTSpaceType = Union{TangentSpaceType,CotangentSpaceType}

"""
    TensorProductType(spaces::VectorSpaceType...)

Vector space type corresponding to the tensor product of given vector space
types.
"""
struct TensorProductType{𝔽,N,TS<:NTuple{N,VectorSpaceType{𝔽}}} <: VectorSpaceType{𝔽}
    spaces::TS
end

"""
    ScalarSpaceType(𝔽)

Vector space of scalars of type 𝔽 (see [`AbstractNumbers`](@ref)).
"""
struct ScalarSpaceType{𝔽} <: VectorSpaceType{𝔽} end

ScalarSpaceType(𝔽) = ScalarSpaceType{𝔽}()

function TensorProductType(spaces::VectorSpaceType{𝔽}...) where {𝔽}
    return TensorProductType{𝔽,length(spaces),typeof(spaces)}(spaces)
end

"""
    AbstractTensorField{𝔽1,TM1<:Manifold,VSIn<:VectorSpaceType,𝔽2,TM2<:Manifold,VSOut<:VectorSpaceType}

An abstract map from vector-valued field over a vector space of type `VSIn` over manifold of
type `TM1` to a vector field over a vector space of type `VSOut` over amnifold of type `TM2`.

!!! note

    Manifold `TM1` doesn't have to be over number system `𝔽1` and manifold `TM2` doesn't
    have to be over number system `𝔽2`
"""
abstract type AbstractTensorField{
    𝔽1,
    TM1<:Manifold,
    VSIn<:VectorSpaceType{𝔽1},
    𝔽2,
    TM2<:Manifold,
    VSOut<:VectorSpaceType{𝔽2},
} end

"""
    apply_operator(op::AbstractTensorField, p, v...)

Apply operator `op` at point `p` to arguments (vectors) `v...`.
"""
function apply_operator(op::AbstractTensorField, p, v...) end

const AbstractScalarValuedField{𝔽,TM,VSIn} = AbstractTensorField{
    𝔽,
    TM,
    VSIn,
    𝔽,
    TM,
    ScalarSpaceType,
} where {𝔽,TM<:Manifold,VSIn<:VectorSpaceType{𝔽}}

"""
    MetricField(M::Manifold)

Multilinear scalar field corresponding to the metric of given manifold `M`.
"""
struct MetricField{𝔽,TM<:Manifold} <: AbstractScalarValuedField{
    𝔽,
    TM,
    TensorProductType{Tuple{TangentSpaceType{𝔽},TangentSpaceType{𝔽}}},
}
    manifold::TM
end

apply_operator(op::MetricField, p, X1, X2) = inner(op.manifold, p, X1, X2)

"""
    AbstractCotangentVectorField{𝔽,TM<:Manifold} <:
        AbstractScalarValuedField{𝔽,TM,TangentSpaceType{𝔽}}

Defines unique representation of cotangent vectors at each point on a manifold.
"""
abstract type AbstractCotangentVectorField{𝔽,TM<:Manifold} <:
              AbstractScalarValuedField{𝔽,TM,TangentSpaceType{𝔽}} end

"""
    RieszRepresenterCotangentVectorField(M::Manifold)

Defines Riesz representer representation of cotangent vectors.
"""
struct RieszRepresenterCotangentVectorField{𝔽,TM<:Manifold} <:
       AbstractCotangentVectorField{𝔽,TM}
    manifold::TM
end

function apply_operator(op::RieszRepresenterCotangentVectorField, p, X1, X2)
    return inner(op.manifold, p, X1, X2)
end

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
struct PushforwardField{𝔽1,TM1<:Manifold,𝔽2,TM2<:Manifold,TF} <:
       AbstractTensorField{𝔽1,TM1,TangentSpaceType{𝔽1},𝔽2,TM2,TangentSpaceType{𝔽2}}
    manifold_in::TM1
    manifold_out::TM2
    f::TF
end

# TODO: get_coordinates for `PushforwardField`

function apply_operator(op::PushforwardField, p, X)
    # TODO
end
