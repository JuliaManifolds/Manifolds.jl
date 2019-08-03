
"""
    VectorSpaceType

Abstract type for tangent spaces, cotangent spaces, their tensor products,
exterior products, etc.

Every vector space `VS` is supposed to provide:
* a method of constructing vectors,
* basic operations: addition, subtraction, multiplication by a scalar
  and negation (unary minus),
* [`zero_vector!(VS, v, x)`](@ref) to construct zero vectors at point `x`,
* `similar(v, T)` for vector `v`,
* `copyto!(v, w)` for vectors `v` and `w`,
* `eltype(v)` for vector `v`,
* [`vector_space_dimension(::VectorBundleFibers{<:typeof(VS)}) where VS`](@ref).

Optionally:
* inner product via `inner` (used to provide Riemannian metric on vector
  bundles),
* [`flat!`](@ref) and [`sharp!`](@ref),
* `norm` (by default uses `inner`),
* [`project_vector!`](@ref) (for embedded vector spaces),
* [`representation_size`](@ref) (if support for [`ProductArray`](@ref) is desired),
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
    VectorSpaceAtPoint(fiber::VectorBundleFibers, x)

A vector space (fiber type `fiber` of a vector bundle) at point `x` from
the manifold `fiber.M`.
"""
struct VectorSpaceAtPoint{TFiber<:VectorBundleFibers, TX}
    fiber::TFiber
    x::TX
end

"""
    TangentSpaceAtPoint(M::Manifold, x)

Return an object of type [`VectorSpaceAtPoint`](@ref) representing tangent
space at `x`.
"""
function TangentSpaceAtPoint(M::Manifold, x)
    return VectorSpaceAtPoint(TangentBundleFibers(M), x)
end

"""
    CotangentSpaceAtPoint(M::Manifold, x)

Return an object of type [`VectorSpaceAtPoint`](@ref) representing cotangent
space at `x`.
"""
function CotangentSpaceAtPoint(M::Manifold, x)
    return VectorSpaceAtPoint(CotangentBundleFibers(M), x)
end

"""
    FVector(type::VectorSpaceType, data)

Decorator indicating that the vector `data` from a fiber of a vector bundle
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
    flat!(M::Manifold, v::FVector, x, w::FVector)

Compute the flat isomorphism (one of the musical isomorphisms) of vector `w`
from the vector space of type `M` at point `x` from manifold `M.M`.

The function can be used for example to transform vectors
from the tangent bundle to vectors from the cotangent bundle
$\flat \colon TM \to T^{*}M$
"""
function flat!(M::Manifold, v::FVector, x, w::FVector)
    error("flat! not implemented for vector bundle fibers space " *
        "of type $(typeof(M)), vector of type $(typeof(v)), point of " *
        "type $(typeof(x)) and vector of type $(typeof(w)).")
end

function flat(M::Manifold, x, w::FVector)
    v = similar_result(M, flat, w, x)
    flat!(M, v, x, w)
    return v
end

function similar_result(M::Manifold, ::typeof(flat), w::FVector{TangentSpaceType}, x)
    return FVector(CotangentSpace, similar(w.data))
end

@doc doc"""
    sharp!(M::Manifold, v::FVector, x, w::FVector)

Compute the sharp isomorphism (one of the musical isomorphisms) of vector `w`
from the vector space of type `M` at point `x` from manifold `M.M`.

The function can be used for example to transform vectors
from the cotangent bundle to vectors from the tangent bundle
$\sharp \colon T^{*}M \to TM$
"""
function sharp!(M::Manifold, v::FVector, x, w::FVector)
    error("sharp! not implemented for vector bundle fibers space " *
        "of type $(typeof(M)), vector of type $(typeof(v)), point of " *
        "type $(typeof(x)) and vector of type $(typeof(w)).")
end

function sharp(M::Manifold, x, w::FVector)
    v = similar_result(M, sharp, w, x)
    sharp!(M, v, x, w)
    return v
end

function similar_result(M::Manifold, ::typeof(sharp), w::FVector{CotangentSpaceType}, x)
    return FVector(TangentSpace, similar(w.data))
end

"""
    similar_result_type(B::VectorBundleFibers, f, args::NTuple{N,Any}) where N

Returns type of element of the array that will represent the result of
function `f` for representing an operation with result in the vector space `VS`
for manifold `M` on given arguments (passed at a tuple).
"""
function similar_result_type(B::VectorBundleFibers, f, args::NTuple{N,Any}) where N
    T = typeof(reduce(+, one(eltype(eti)) for eti ∈ args))
    return T
end

"""
    similar_result(B::VectorBundleFibers, f, x...)

Allocates an array for the result of function `f` that is
an element of the vector space of type `B.VS` on manifold `B.M`
and arguments `x...` for implementing the non-modifying operation
using the modifying operation.
"""
function similar_result(B::VectorBundleFibers, f, x...)
    T = similar_result_type(B, f, x)
    return similar(x[1], T)
end

"""
    distance(B::VectorBundleFibers, x, v, w)

Distance between vectors `v` and `w` from the vector space at point `x`
from the manifold `M.M`, that is the base manifold of `M`.
"""
distance(B::VectorBundleFibers, x, v, w) = norm(B, x, v-w)

"""
    inner(B::VectorBundleFibers, x, v, w)

Inner product of vectors `v` and `w` from the vector space of type `B.VS`
at point `x` from manifold `B.M`.
"""
function inner(B::VectorBundleFibers, x, v, w)
    error("inner not defined for vector space family of type $(typeof(B)), " *
        "point of type $(typeof(x)) and " *
        "vectors of types $(typeof(v)) and $(typeof(w)).")
end

function inner(B::VectorBundleFibers{<:TangentSpaceType}, x, v, w)
    return inner(B.M, x, v, w)
end

function inner(B::VectorBundleFibers{<:CotangentSpaceType}, x, v, w)
    return inner(B.M,
                 x,
                 sharp(B.M, x, FVector(CotangentSpace, v)).data,
                 sharp(B.M, x, FVector(CotangentSpace, w)).data)
end

norm(B::VectorBundleFibers, x, v) = sqrt(inner(B, x, v, v))

norm(B::VectorBundleFibers{<:TangentSpaceType}, x, v) = norm(B.M, x, v)

"""
    vector_space_dimension(B::VectorBundleFibers)

Dimension of the vector space of type `B`.
"""
function vector_space_dimension(B::VectorBundleFibers)
    error("vector_space_dimension not implemented for vector space family $(typeof(B)).")
end

vector_space_dimension(B::VectorBundleFibers{<:TCoTSpaceType}) = manifold_dimension(B.M)

function vector_space_dimension(B::VectorBundleFibers{<:TensorProductType})
    dim = 1
    for space in B.VS.spaces
        dim *= vector_space_dimension(VectorBundleFibers(space, B.M))
    end
    return dim
end

function representation_size(B::VectorBundleFibers{<:TCoTSpaceType})
    representation_size(B.M)
end

"""
    zero_vector!(B::VectorBundleFibers, v, x)

Save the zero vector from the vector space of type `B.VS` at point `x`
from manifold `B.M` to `v`.
"""
function zero_vector!(B::VectorBundleFibers, v, x)
    error("zero_vector! not implemented for vector space family of type $(typeof(B)).")
end

function zero_vector!(B::VectorBundleFibers{<:TangentSpaceType}, v, x)
    zero_tangent_vector!(B.M, v, x)
    return v
end

"""
    zero_vector(B::VectorBundleFibers, x)

Compute the zero vector from the vector space of type `B.VS` at point `x`
from manifold `B.M`.
"""
function zero_vector(B::VectorBundleFibers, x)
    v = similar_result(B, zero_vector, x)
    zero_vector!(B, v, x)
    return v
end

"""
    project_vector!(B::VectorBundleFibers, v, x, w)

Project vector `w` from the vector space of type `B.VS` at point `x`
and save the result to `v`.
"""
function project_vector!(B::VectorBundleFibers, v, x, w)
    error("project_vector! not implemented for vector space family of type $(typeof(B)), output vector of type $(typeof(v)) and input vector at point $(typeof(x)) with type of w $(typeof(w)).")
end

function project_vector!(B::VectorBundleFibers{<:TangentSpaceType}, v, x, w)
    project_tangent!(B.M, v, x, w)
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

function VectorBundle(VS::TVS, M::TM) where {TVS<:VectorSpaceType, TM<:Manifold}
    return VectorBundle{TVS, TM}(VS, M, VectorBundleFibers(VS, M))
end

function representation_size(B::VectorBundle)
    len_manifold = prod(representation_size(B.M))
    len_vs = prod(representation_size(B.VS))
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
    bundle_projection(B::VectorBundle, x::ProductRepr)

Projection of point `x` from the bundle `M` to the base manifold.
Returns the point on the base manifold `B.M` at which the vector part
of `x` is attached.
"""
function bundle_projection(B::VectorBundle, x)
    return submanifold_component(x, 1)
end

function isapprox(B::VectorBundle, x, y; kwargs...)
    return isapprox(B.M, x.parts[1], y.parts[1]; kwargs...) &&
        isapprox(x.parts[2], y.parts[2]; kwargs...)
end

function isapprox(B::VectorBundle, x, v, w; kwargs...)
    return isapprox(B.M, v.parts[1], w.parts[1]; kwargs...) &&
        isapprox(B.M, x.parts[1], v.parts[2], w.parts[2]; kwargs...)
end

manifold_dimension(B::VectorBundle) = manifold_dimension(B.M) + vector_space_dimension(B.VS)

function inner(B::VectorBundle, x, v, w)
    return inner(B.M, x.parts[1], v.parts[1], w.parts[1]) +
           inner(B.VS, x.parts[2], v.parts[2], w.parts[2])
end

function distance(B::VectorBundle, x, y)
    dist_man = distance(B.M, x.parts[1], y.parts[1])
    vy_x = vector_transport_to(B.M, y.parts[1], y.parts[2], x.parts[1])
    dist_vec = distance(B.VS, x.parts[1], x.parts[2], vy_x)

    return sqrt(dist_man^2 + dist_vec^2)
end

function exp!(B::VectorBundle, y, x, v)
    exp!(B.M, y.parts[1], x.parts[1], v.parts[1])
    vector_transport_to!(B.M, y.parts[2], x.parts[1], x.parts[2] + v.parts[2], y.parts[1])
    return y
end

function log!(B::VectorBundle, v, x, y)
    log!(B.M, v.parts[1], x.parts[1], y.parts[1])
    vector_transport_to!(B.M, v.parts[2], y.parts[1], y.parts[2], x.parts[1])
    copyto!(v.parts[2], v.parts[2] - x.parts[2])
    return v
end

function zero_tangent_vector!(B::VectorBundle, v, x)
    zero_tangent_vector!(B.M, v.parts[1], x.parts[1])
    zero_vector!(B.VS, v.parts[2], x.parts[2])
    return v
end

function project_point!(B::VectorBundle, x)
    project_point!(B.M, x.parts[1])
    project_tangent!(B.M, x.parts[2], x.parts[1], x.parts[2])
    return x
end

function project_tangent!(B::VectorBundle, w, x, v)
    project_tangent!(B.M, w.parts[1], x.parts[1], v.parts[1])
    project_tangent!(B.M, w.parts[2], x.parts[1], v.parts[2])
    return w
end
