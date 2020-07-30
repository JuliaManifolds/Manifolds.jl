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
abstract type VectorSpaceType end

struct TangentSpaceType <: VectorSpaceType end

struct CotangentSpaceType <: VectorSpaceType end

TCoTSpaceType = Union{TangentSpaceType,CotangentSpaceType}

const TangentSpace = TangentSpaceType()
const CotangentSpace = CotangentSpaceType()

"""
    TensorProductType(spaces::VectorSpaceType...)

Vector space type corresponding to the tensor product of given vector space
types.
"""
struct TensorProductType{TS<:Tuple} <: VectorSpaceType
    spaces::TS
end

"""
    ScalarSpaceType()

Vector space of scalars.
"""
struct ScalarSpaceType <: VectorSpaceType end

TensorProductType(spaces::VectorSpaceType...) = TensorProductType{typeof(spaces)}(spaces)

"""
    AbstractTensorField{𝔽1,TM1<:Manifold{𝔽1},VSIn<:VectorSpaceType,𝔽2,TM2<:Manifold{𝔽2},VSOut<:VectorSpaceType}

An abstract map from vector-valued field over a vector space of type `VSIn` over manifold of
type `TM1` to a vector field over a vector space of type `VSOut` over amnifold of type `TM2`.
"""
abstract type AbstractTensorField{
    𝔽1,
    TM1<:Manifold{𝔽1},
    VSIn<:VectorSpaceType,
    𝔽2,
    TM2<:Manifold{𝔽2},
    VSOut<:VectorSpaceType,
} end

"""
    apply_operator(op::AbstractTensorField, p, v...)

Apply operator `op` at point `p` to arguments (vectors) `v...`.
"""
function apply_operator(op::AbstractTensorField, p, v...)
    Y = allocate_result(F, apply_operator, p, v...)
    return apply_operator!(F, Y, p, v...)
end

const AbstractScalarValuedField{𝔽,TM,VSIn} = AbstractTensorField{
    𝔽,
    TM,
    VSIn,
    𝔽,
    TM,
    ScalarSpaceType,
} where {𝔽,TM<:Manifold{𝔽},VSIn<:VectorSpaceType}

"""
    MetricField(M::Manifold)

Multilinear scalar field corresponding to the metric of given manifold `M`.
"""
struct MetricField{𝔽,TM<:Manifold{𝔽}} <: AbstractScalarValuedField{
    𝔽,
    TM,
    TensorProductType{Tuple{TangentSpaceType,TangentSpaceType}},
}
    manifold::TM
end

apply_operator(op::MetricField, p, X1, X2) = inner(op.manifold, p, X1, X2)

"""
    AbstractCotangentVectorField{𝔽,TM<:Manifold{𝔽}} <: AbstractScalarValuedField{𝔽,TM,TangentSpaceType}

Defines unique representation of cotangent vectors at each point on a manifold.
"""
abstract type AbstractCotangentVectorField{𝔽,TM<:Manifold{𝔽}} <:
              AbstractScalarValuedField{𝔽,TM,TangentSpaceType} end

"""
    RieszRepresenterCotangentVectorField(M::Manifold)

Defines Riesz representer representation of cotangent vectors. By default, any tangent
vector space basis defines a contangent vector space basis in this representation by
expanding the Riesz representer.
"""
struct RieszRepresenterCotangentVectorField{𝔽,TM<:Manifold{𝔽}} <:
       AbstractCotangentVectorField{𝔽,TM}
    manifold::TM
end

function apply_operator(op::RieszRepresenterCotangentVectorField, p, X1, X2)
    return inner(op.manifold, p, X1, X2)
end

function get_coordinates!(
    F::RieszRepresenterCotangentVectorField,
    Y,
    p,
    X,
    B::AbstractBasis,
)
    return get_coordinates!(F.manifold, Y, p, X, B)
end

function get_coordinates(F::AbstractTensorField, p, X, B::AbstractBasis)
    Y = allocate_result(F, get_coordinates, p, X, B)
    return get_coordinates!(F, Y, p, X, B)
end

"""
    get_cotangent_operator(M::Manifold)

Get the default representation of cotangent vectors for manifold `M`. Defaults to
[`RieszRepresenterCotangentVectorField`](@ref).
"""
function get_cotangent_operator(M::Manifold)
    return RieszRepresenterCotangentVectorField(M)
end


abstract type AbstractAffineConnection{𝔽,TM<:Manifold{𝔽}} <: AbstractTensorField{
    𝔽,
    TM,
    TensorProductType{Tuple{TangentSpaceType,TangentSpaceType}},
    𝔽,
    TM,
    TangentSpaceType,
} end


"""
    LeviCivitaConnection{𝔽,TM<:Manifold{𝔽}} <: AbstractAffineConnection{𝔽,TM}

Represents the Levi-Civita connection on manifold `TM`.
"""
abstract type LeviCivitaConnection{𝔽,TM<:Manifold{𝔽}} <: AbstractAffineConnection{𝔽,TM}
end

"""
    FunctionLeviCivitaConnection{𝔽,TM<:Manifold{𝔽},TF} <: LeviCivitaConnection{𝔽,TM}

Levi-Civita connection of a tangent vector valued function `f` defined on manifold `M`.


"""
struct FunctionLeviCivitaConnection{𝔽,TM<:Manifold{𝔽},TF} <: LeviCivitaConnection{𝔽,TM}
    manifold::TM
    f::TF
end

@doc raw"""
    apply_operator(F::LeviCivitaConnection, p, X)

Compute the value of the Levi-Civita connection at point `p`, in the direction pointed by
tangent vector `X` at `p`, of the vector field on `M.manifold` defined by a function `fY`
defined by `F`.
The formula reads $(\nabla_X \mathit{fY})_p$.
"""
apply_operator(F::LeviCivitaConnection, p, X)

"""
    apply_operator(
        F::FunctionLeviCivitaConnection{𝔽,<:AbstractEmbeddedManifold{DefaultIsometricEmbeddingType}},
        p,
        X,
    ) where {𝔽}

Apply the Levi-Civita connection on an isometrically embedded manifold by applying the
connection in the embedding and projecting it back.

See [^Absil2008], Section 5.3.3 for details.

[^Absil2008]:
    > Absil, P. A., et al. Optimization Algorithms on Matrix Manifolds. 2008.
"""
function apply_operator(
    F::FunctionLeviCivitaConnection{𝔽,<:AbstractEmbeddedManifold{DefaultIsometricEmbeddingType}},
    p,
    X,
) where {𝔽}
    emb_Z = apply_operator(
        FunctionLeviCivitaConnection(base_manifold(F.manifold), q -> embed(F.manifold, q, F.f(q))),
        embed(F.manifold, p),
        embed(F.manifold, p, X),
    )
    return project(F.manifold, p, emb_Z)
end

function apply_operator!(
    F::FunctionLeviCivitaConnection{𝔽,<:AbstractEmbeddedManifold{DefaultIsometricEmbeddingType}},
    Z,
    p,
    X,
) where {𝔽}
    emb_Z = apply_operator(
        FunctionLeviCivitaConnection(base_manifold(F.manifold), q -> embed(F.manifold, q, F.f(q))),
        Z,
        embed(L.manifold, p),
        embed(L.manifold, p, X),
    )
    return project!(F.manifold, Z, p, emb_Z)
end

"""
    PushforwardField(manifold_in::Manifold, manifold_out::Manifold, f)

Pushforward of function `f` taking arguments in `manifold_in` and values in `manifold_out`.
"""
struct PushforwardField{𝔽1,TM1<:Manifold{𝔽1},𝔽2,TM2<:Manifold{𝔽2},TF} <:
       AbstractTensorField{𝔽1,TM1,TangentSpaceType,𝔽2,TM2,TangentSpaceType}
    manifold_in::TM1
    manifold_out::TM2
    f::TF
end

# TODO: get_coordinates for `PushforwardField`

function apply_operator(op::PushforwardField, p, X)
    # TODO
end
