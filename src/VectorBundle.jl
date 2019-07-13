
"""
    VectorSpaceType

Abstract type for tangent space, cotangent space, their tensor products,
exterior products etc.
"""
abstract type VectorSpaceType end

struct TangentSpaceType <: VectorSpaceType end
struct CotangentSpaceType <: VectorSpaceType end

TCoTSpaceType = Union{TangentSpaceType, CotangentSpaceType}

const TangentSpace = TangentSpaceType()
const CotangentSpace = CotangentSpaceType()

"""
    ManifoldVectorSpace(VS::VectorSpaceType, M::Manifold)

Type representing a family of vector spaces of the vector bundle over `M`
with vectors spaces of type `M`. In contrast with `VectorBundle`, operations
on `ManifoldVectorSpace` expect point-like and vector-like parts to be
passed separately instead of being bundled together. It can be thought of
as a representation of vector spaces from a vector bundle but without
storing the point at which a vector space is attached (which is specified
separately in various functions).
"""
struct ManifoldVectorSpace{TVS<:VectorSpaceType, TM<:Manifold}
    VS::TVS
    M::TM
end

@doc doc"""
    flat_isomorphism!(M::ManifoldVectorSpace, v, x, w)

Compute the flat isomorphism (one of the musical isomorphisms) of vector `w`
from the vector space of type `M` at point `x` from manifold `M.M`.

The function can be used for example to transform vectors
from the tangent bundle to vectors from the cotangent bundle
$\flat \colon TM \to T^{*}M$
"""
function flat_isomorphism!(M::ManifoldVectorSpace, v, x, w)
    error("flat_isomorphism! not implemented for manifold vector space
        of type $(typeof(M)), vector of type $(typeof(v)), point of " *
        "type $(typeof(x)) and vector of type $(typeof(w)).")
end

function flat_isomorphism(M::ManifoldVectorSpace, x, w)
    v = similar_result(M, flat_isomorphism, w, x)
    flat_isomorphism!(M, v, x, w)
    return v
end

"""
    similar_result_type(M::ManifoldVectorSpace, f, args::NTuple{N,Any}) where N

Returns type of element of the array that will represent the result of
function `f` for representing an operation with result in the vector space `VS`
for manifold `M` on given arguments (passed at a tuple).
"""
function similar_result_type(M::ManifoldVectorSpace, f, args::NTuple{N,Any}) where N
    T = typeof(reduce(+, one(eltype(eti)) for eti ∈ args))
    return T
end

"""
    similar_result(M::ManifoldVectorSpace, f, x...)

Allocates an array for the result of function `f` that is
an element of the vector space of type `M.VS` on manifold `M.M`
and arguments `x...` for implementing the non-modifying operation
using the modifying operation.
"""
function similar_result(M::ManifoldVectorSpace, f, x...)
    T = similar_result_type(M, f, x)
    return similar(x[1], T)
end

norm(M::ManifoldVectorSpace{<:TCoTSpaceType}, x, v) = norm(v)

"""
    inner(M::ManifoldVectorSpace, x, v, w)

Inner product of vectors `v` and `w` from the vector space of type `VS`
at point `x` from manifold `M`.
"""
function inner(M::ManifoldVectorSpace, x, v, w)
    error("inner not defined for vector space family of type $(typeof(M)), " *
        "point of type $(typeof(x)) and " *
        "vectors of types $(typeof(v)) and $(typeof(w)).")
end

function inner(M::ManifoldVectorSpace{<:TangentSpaceType}, x, v, w)
    return inner(M.M, x, v, w)
end

function inner(M::ManifoldVectorSpace{<:CotangentSpaceType}, x, v, w)
    return inner(M.M, x, flat_isomorphism(M, x, v), flat_isomorphism(M, x, w))
end

"""
    vector_space_dimension(M::ManifoldVectorSpace)

Dimension of the vector space of type `M`.
"""
function vector_space_dimension(M::ManifoldVectorSpace)
    error("vector_space_dimension not implemented for vector space family $(typeof(M)).")
end

vector_space_dimension(M::ManifoldVectorSpace{<:TCoTSpaceType}) = manifold_dimension(M.M)

function representation_size(M::ManifoldVectorSpace{<:TCoTSpaceType})
    representation_size(M.M)
end

"""
    zero_vector!(M::ManifoldVectorSpace, v, x)

Save the zero vector from the vector space of type `VS` at point `x`
from manifold `M` to `v`.
"""
function zero_vector!(M::ManifoldVectorSpace, v, x)
    error("zero_vector! not implemented for manifold $(typeof(M)), vector space of type $(typeof(VS)), vector of type $(typeof(v)) and point of type $(typeof(x)).")
end

function zero_vector!(M::ManifoldVectorSpace{<:TangentSpaceType}, v, x)
    zero_tangent_vector!(M.M, v, x)
    return v
end

"""
    zero_vector(M::ManifoldVectorSpace, x)

Compute the zero vector from the vector space of type `VS` at point `x`
from manifold `M.M`.
"""
function zero_vector(M::ManifoldVectorSpace, x)
    v = similar_result(M, zero_vector, x)
    zero_vector!(M, v, x)
    return v
end

"""
    project_vector!(M::ManifoldVectorSpace, v, x, w)

Project vector `w` from the vector space of type `VS` at point `x`
and save the result to `v`.
"""
function project_vector!(M::ManifoldVectorSpace, v, x, w)
    error("project_vector! not implemented for vector space manifold $(typeof(M)), vector space of type $(typeof(VS)), output vector of type $(typeof(v)) and input vector at point $(typeof(x)) with type of w $(typeof(w)).")
end

function project_vector!(M::ManifoldVectorSpace{<:TangentSpaceType}, v, x, w)
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
    VS::ManifoldVectorSpace{TVS, TM}
end

function VectorBundle(VS::TVS, M::TM) where{TVS<:VectorSpaceType, TM<:Manifold}
    return VectorBundle{TVS, TM}(VS, M, ManifoldVectorSpace(VS, M))
end

function representation_size(M::VectorBundle)
    len_manifold = prod(representation_size(M.M))
    len_vs = prod(representation_size(M.VS))
    return (len_manifold + len_vs,)
end

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


"""
    vector_distance(M::VectorBundle, x, v, w)

Distance between vectors `v` and `w` from the tangent space at point `x`
from the manifold `M.M`, that is the base manifold of bundle `M`.
"""
function vector_distance(M::VectorBundle, x, v, w)
    error("vector_distance not defined for vector bundle of type $(typeof(M)), " *
        "point of type $(typeof(x)) and vectors of types $(typeof(v)) " *
        "and $(typeof(w)).")
end

vector_distance(M::VectorBundle{<:TCoTSpaceType}, x, v, w) = norm(v-w)

function distance(M::VectorBundle, x, y)
    dist_man = distance(M.M, x.parts[1], y.parts[1])
    vy_x = vector_transport(M.M, y.parts[1], y.parts[2], x.parts[1])
    dist_vec = vector_distance(M, x.parts[1], x.parts[2], vy_x)

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
