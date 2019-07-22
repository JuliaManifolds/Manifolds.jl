
"""
    VectorSpaceType

Abstract type for tangent space, cotangent space, their tensor products,
exterior products etc.

Every vector space `VS` is supposed to provide:
* a method of constucting vectors,
* basic operations: addition, subtraction, multiplication by a scalar
  and negation (unary minus),
* zero_vector!(VS, v, x) to construct zero vectors at point `x`,
* `similar(v, T)` for vector `v`,
* `copyto!(v, w)` for vectors `v` and `w`,
* `eltype(v)` for vector `v`,
* `vector_space_dimension(::VectorBundleFibers{<:typeof(VS)})`.

Optionally:
* inner product via `inner` (used to provide Riemannian metric on vector
  bundles),
* flat_isomorphism! and sharp_isomorphism!,
* norm (by default uses `inner`),
* project_vector! (for embedded vector spaces),
* representation_size (if support for `ProductArray` is desired),
* broadcasting for basic operations.
"""
abstract type VectorSpaceType end

struct TangentSpaceType <: VectorSpaceType end
struct CotangentSpaceType <: VectorSpaceType end

TCoTSpaceType = Union{TangentSpaceType, CotangentSpaceType}

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

TensorProductType(spaces::VectorSpaceType...) = TensorProductType{typeof(spaces)}(spaces)

"""
    VectorBundleFibers(VS::VectorSpaceType, M::Manifold)

Type representing a family of vector spaces (fibers) of a vector bundle over `M`
with vector spaces of type `VS`. In contrast with `VectorBundle`, operations
on `VectorBundleFibers` expect point-like and vector-like parts to be
passed separately instead of being bundled together. It can be thought of
as a representation of vector spaces from a vector bundle but without
storing the point at which a vector space is attached (which is specified
separately in various functions).
"""
struct VectorBundleFibers{TVS<:VectorSpaceType, TM<:Manifold}
    VS::TVS
    M::TM
end

"""
    FVector(x, type::VectorSpaceType)

Decorator indicating that the vector `x` from a fiber of a tangent bundle
is of type `type`.
"""
struct FVector{TType<:VectorSpaceType,TData}
    type::TType
    data::TData
end

size(x::FVector) = size(x.data)
Base.@propagate_inbounds getindex(x::FVector, i) = getindex(x.data, i)
Base.@propagate_inbounds setindex!(x::FVector, val, i) = setindex!(x.data, val, i)

(+)(v1::FVector, v2::FVector) = FVector(v1.type, v1.data + v2.data)
(-)(v1::FVector, v2::FVector) = FVector(v1.type, v1.data - v2.data)
(-)(v::FVector) = FVector(v.type, -v.data)
(*)(a::Number, v::FVector) = FVector(v.type, a*v.data)

eltype(::Type{FVector{TType,TData}}) where {TType<:VectorSpaceType,TData} = eltype(TData)
similar(x::FVector) = FVector(x.type, similar(x.data))
similar(x::FVector, ::Type{T}) where {T} = FVector(x.type, similar(x.data, T))

@doc doc"""
    flat_isomorphism!(M::Manifold, v::FVector, x, w::FVector)

Compute the flat isomorphism (one of the musical isomorphisms) of vector `w`
from the vector space of type `M` at point `x` from manifold `M.M`.

The function can be used for example to transform vectors
from the tangent bundle to vectors from the cotangent bundle
$\flat \colon TM \to T^{*}M$
"""
function flat_isomorphism!(M::Manifold, v::FVector, x, w::FVector)
    error("flat_isomorphism! not implemented for vector bundle fibers space " *
        "of type $(typeof(M)), vector of type $(typeof(v)), point of " *
        "type $(typeof(x)) and vector of type $(typeof(w)).")
end

function flat_isomorphism(M::Manifold, x, w::FVector)
    v = similar_result(M, flat_isomorphism, w, x)
    flat_isomorphism!(M, v, x, w)
    return v
end

function similar_result(M::Manifold, ::typeof(flat_isomorphism), w::FVector{TangentSpaceType}, x)
    return FVector(CotangentSpace, similar(w.data))
end

@doc doc"""
    sharp_isomorphism!(M::Manifold, v::FVector, x, w::FVector)

Compute the sharp isomorphism (one of the musical isomorphisms) of vector `w`
from the vector space of type `M` at point `x` from manifold `M.M`.

The function can be used for example to transform vectors
from the cotangent bundle to vectors from the tangent bundle
$\sharp \colon T^{*}M \to TM$
"""
function sharp_isomorphism!(M::Manifold, v::FVector, x, w::FVector)
    error("sharp_isomorphism! not implemented for vector bundle fibers space " *
        "of type $(typeof(M)), vector of type $(typeof(v)), point of " *
        "type $(typeof(x)) and vector of type $(typeof(w)).")
end

function sharp_isomorphism(M::Manifold, x, w::FVector)
    v = similar_result(M, sharp_isomorphism, w, x)
    sharp_isomorphism!(M, v, x, w)
    return v
end

function similar_result(M::Manifold, ::typeof(sharp_isomorphism), w::FVector{CotangentSpaceType}, x)
    return FVector(TangentSpace, similar(w.data))
end

"""
    similar_result_type(M::VectorBundleFibers, f, args::NTuple{N,Any}) where N

Returns type of element of the array that will represent the result of
function `f` for representing an operation with result in the vector space `VS`
for manifold `M` on given arguments (passed at a tuple).
"""
function similar_result_type(M::VectorBundleFibers, f, args::NTuple{N,Any}) where N
    T = typeof(reduce(+, one(eltype(eti)) for eti ∈ args))
    return T
end

"""
    similar_result(M::VectorBundleFibers, f, x...)

Allocates an array for the result of function `f` that is
an element of the vector space of type `M.VS` on manifold `M.M`
and arguments `x...` for implementing the non-modifying operation
using the modifying operation.
"""
function similar_result(M::VectorBundleFibers, f, x...)
    T = similar_result_type(M, f, x)
    return similar(x[1], T)
end

norm(M::VectorBundleFibers, x, v) = sqrt(inner(M, x, v, v))

norm(M::VectorBundleFibers{<:TangentSpaceType}, x, v) = norm(M.M, x, v)

"""
    vector_distance(M::VectorBundleFibers, x, v, w)

Distance between vectors `v` and `w` from the vector space at point `x`
from the manifold `M.M`, that is the base manifold of `M`.
"""
vector_distance(M::VectorBundleFibers, x, v, w) = norm(M, x, v-w)

"""
    inner(M::VectorBundleFibers, x, v, w)

Inner product of vectors `v` and `w` from the vector space of type `VS`
at point `x` from manifold `M`.
"""
function inner(M::VectorBundleFibers, x, v, w)
    error("inner not defined for vector space family of type $(typeof(M)), " *
        "point of type $(typeof(x)) and " *
        "vectors of types $(typeof(v)) and $(typeof(w)).")
end

function inner(M::VectorBundleFibers{<:TangentSpaceType}, x, v, w)
    return inner(M.M, x, v, w)
end

function inner(M::VectorBundleFibers{<:CotangentSpaceType}, x, v, w)
    return inner(M.M, x, flat_isomorphism(M, x, v), flat_isomorphism(M, x, w))
end

"""
    vector_space_dimension(M::VectorBundleFibers)

Dimension of the vector space of type `M`.
"""
function vector_space_dimension(M::VectorBundleFibers)
    error("vector_space_dimension not implemented for vector space family $(typeof(M)).")
end

vector_space_dimension(M::VectorBundleFibers{<:TCoTSpaceType}) = manifold_dimension(M.M)

function vector_space_dimension(M::VectorBundleFibers{<:TensorProductType})
    dim = 1
    for space in M.VS.spaces
        dim *= vector_space_dimension(VectorBundleFibers(space, M.M))
    end
    return dim
end

function representation_size(M::VectorBundleFibers{<:TCoTSpaceType})
    representation_size(M.M)
end

"""
    zero_vector!(M::VectorBundleFibers, v, x)

Save the zero vector from the vector space of type `VS` at point `x`
from manifold `M` to `v`.
"""
function zero_vector!(M::VectorBundleFibers, v, x)
    error("zero_vector! not implemented for manifold $(typeof(M)), vector space of type $(typeof(VS)), vector of type $(typeof(v)) and point of type $(typeof(x)).")
end

function zero_vector!(M::VectorBundleFibers{<:TangentSpaceType}, v, x)
    zero_tangent_vector!(M.M, v, x)
    return v
end

"""
    zero_vector(M::VectorBundleFibers, x)

Compute the zero vector from the vector space of type `VS` at point `x`
from manifold `M.M`.
"""
function zero_vector(M::VectorBundleFibers, x)
    v = similar_result(M, zero_vector, x)
    zero_vector!(M, v, x)
    return v
end

"""
    project_vector!(M::VectorBundleFibers, v, x, w)

Project vector `w` from the vector space of type `VS` at point `x`
and save the result to `v`.
"""
function project_vector!(M::VectorBundleFibers, v, x, w)
    error("project_vector! not implemented for vector space manifold $(typeof(M)), vector space of type $(typeof(VS)), output vector of type $(typeof(v)) and input vector at point $(typeof(x)) with type of w $(typeof(w)).")
end

function project_vector!(M::VectorBundleFibers{<:TangentSpaceType}, v, x, w)
    project_tangent!(M, v, x, w)
    return v
end

"""
    VectorBundle(M::Manifold, type::VectorSpaceType)

Vector bundle on manifold `M` of type `type`.
"""
struct VectorBundle{TVS<:VectorSpaceType, TM<:Manifold} <: Manifold
    type::TVS
    M::TM
    VS::VectorBundleFibers{TVS, TM}
end

function VectorBundle(VS::TVS, M::TM) where{TVS<:VectorSpaceType, TM<:Manifold}
    return VectorBundle{TVS, TM}(VS, M, VectorBundleFibers(VS, M))
end

function representation_size(M::VectorBundle)
    len_manifold = prod(representation_size(M.M))
    len_vs = prod(representation_size(M.VS))
    return (len_manifold + len_vs,)
end

TangentBundleFibers{M} = VectorBundleFibers{TangentSpaceType,M}
TangentBundleFibers(M::Manifold) = VectorBundleFibers(TangentSpace, M)

CotangentBundleFibers{M} = VectorBundleFibers{CotangentSpaceType,M}
CotangentBundleFibers(M::Manifold) = VectorBundleFibers(CotangentSpace, M)

TangentBundle{M} = VectorBundle{TangentSpaceType,M}
TangentBundle(M::Manifold) = VectorBundle(TangentSpace, M)

CotangentBundle{M} = VectorBundle{CotangentSpaceType,M}
CotangentBundle(M::Manifold) = VectorBundle(CotangentSpace, M)

"""
    bundle_projection(M::VectorBundle, x::ProductRepr)

Projection of point `x` from the bundle `M` to the base manifold.
Returns the point on the base manifold `M.M` at which the vector part
of `x` is attached.
"""
function bundle_projection(M::VectorBundle, x)
    return submanifold_component(x, 1)
end

function isapprox(M::VectorBundle, x, y; kwargs...)
    return isapprox(M.M, x.parts[1], y.parts[1]; kwargs...) &&
        isapprox(x.parts[2], y.parts[2]; kwargs...)
end

function isapprox(M::VectorBundle, x, v, w; kwargs...)
    return isapprox(M.M, v.parts[1], w.parts[1]; kwargs...) &&
        isapprox(M.M, x.parts[1], v.parts[2], w.parts[2]; kwargs...)
end

manifold_dimension(M::VectorBundle) = manifold_dimension(M.M) + vector_space_dimension(M.VS)

function inner(M::VectorBundle, x, v, w)
    return inner(M.M, x.parts[1], v.parts[1], w.parts[1]) +
           inner(M.VS, x.parts[2], v.parts[2], w.parts[2])
end

function distance(M::VectorBundle, x, y)
    dist_man = distance(M.M, x.parts[1], y.parts[1])
    vy_x = vector_transport(M.M, y.parts[1], y.parts[2], x.parts[1])
    dist_vec = vector_distance(M.VS, x.parts[1], x.parts[2], vy_x)

    return sqrt(dist_man^2 + dist_vec^2)
end

function exp!(M::VectorBundle, y, x, v)
    exp!(M.M, y.parts[1], x.parts[1], v.parts[1])
    vector_transport!(M.M, y.parts[2], x.parts[1], x.parts[2] + v.parts[2], y.parts[1])
    return y
end

function log!(M::VectorBundle, v, x, y)
    log!(M.M, v.parts[1], x.parts[1], y.parts[1])
    vector_transport!(M.M, v.parts[2], y.parts[1], y.parts[2], x.parts[1])
    copyto!(v.parts[2], v.parts[2] - x.parts[2])
    return v
end

function zero_tangent_vector!(M::VectorBundle, v, x)
    zero_tangent_vector!(M.M, v.parts[1], x.parts[1])
    zero_vector!(M.VS, v.parts[2], x.parts[2])
    return v
end

function project_point!(M::VectorBundle, x)
    project_point!(M.M, x.parts[1])
    project_tangent!(M.M, x.parts[2], x.parts[1], x.parts[2])
    return x
end

function project_tangent!(M::VectorBundle, w, x, v)
    project_tangent!(M.M, w.parts[1], x.parts[1], v.parts[1])
    project_tangent!(M.M, w.parts[2], x.parts[1], v.parts[2])
    return w
end
