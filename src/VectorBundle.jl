
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
    similar_result_type(M::Manifold, VS::VectorSpaceType, f, args::NTuple{N,Any}) where N

Returns type of element of the array that will represent the result of
function `f` for representing an operation with result in the vector space `VS`
for manifold `M` on given arguments (passed at a tuple).
"""
function similar_result_type(M::Manifold, VS::VectorSpaceType, f, args::NTuple{N,Any}) where N
    T = typeof(reduce(+, one(eltype(eti)) for eti ∈ args))
    return T
end

"""
    similar_result(M::Manifold, VS::VectorSpaceType, f, x...)

Allocates an array for the result of function `f` that is
an element of the vector space of type `VS` on manifold `M`
and arguments `x...` for implementing the non-modifying operation
using the modifying operation.
"""
function similar_result(M::Manifold, VS::VectorSpaceType, f, x...)
    T = similar_result_type(M, VS, f, x)
    return similar(x[1], T)
end


"""
    manifold_dimension(M::Manifold, VS::VectorSpaceType)

Dimension of the vector space of type `VS` over manifold `M`.
"""
function manifold_dimension(M::Manifold, VS::VectorSpaceType)
    error("manifold_dimension not implemented for manifold $(typeof(M)) and vector space of type $(typeof(VS)).")
end

manifold_dimension(M::Manifold, ::TCoTSpaceType) = manifold_dimension(M)

function representation_size(M::Manifold, VS::TCoTSpaceType)
    representation_size(M)
end

"""
    zero_vector!(M::Manifold, VS::VectorSpaceType, v, x)

Save the zero vector from the vector space of type `VS` at point `x`
from manifold `M` to `v`.
"""
function zero_vector!(M::Manifold, VS::VectorSpaceType, v, x)
    error("zero_vector! not implemented for manifold $(typeof(M)), vector space of type $(typeof(VS)), vector of type $(typeof(v)) and point of type $(typeof(x)).")
end

function zero_vector!(M::Manifold, VS::TangentSpaceType, v, x)
    zero_tangent_vector!(M, v, x)
    return v
end

"""
    zero_vector(M::Manifold, VS::VectorSpaceType, x)

Compute the zero vector from the vector space of type `VS` at point `x`
from manifold `M`.
"""
function zero_vector(M::Manifold, VS::VectorSpaceType, x)
    v = similar_result(M, VS, zero_vector, x)
    zero_vector!(M, VS, v, x)
    return v
end

"""
    VectorSpaceManifold(M::Manifold, VS::VectorSpaceType, point)

Manifold corresponding to the vector space `VS` (for example tangent space
or cotangent space) at point `point` from manifold `M`.
"""
struct VectorSpaceManifold{TVS<:VectorSpaceType, TM<:Manifold, TP} <: Manifold
    VS::TVS
    M::TM
    point::TP
end

"""
    project_vector!(vs::VectorSpaceManifold, v, w)

Project vector `w` from the vector space `vs` and save the result to `v`.
"""
function project_vector!(M::VectorSpaceManifold, v, w)
    error("project_vector! not implemented for vector space manifold $(typeof(M)) and vectors of type $(typeof(v)) and $(typeof(w)).")
end

function project_vector!(M::VectorSpaceManifold{<:TCoTSpaceType}, v, w)
    project_tangent!(M.M, v, Ms.point, w)
    return v
end

manifold_dimension(M::VectorSpaceManifold) = manifold_dimension(M.M, M.VS)

function representation_size(M::VectorSpaceManifold)
    representation_size(M.M, M.VS)
end

@inline inner(::VectorSpaceManifold{<:TCoTSpaceType}, x, v, w) = dot(v, w)

distance(::VectorSpaceManifold, x, y) = norm(x-y)
norm(::VectorSpaceManifold{TCoTSpaceType}, x, v) = norm(v)

exp!(M::VectorSpaceManifold, y, x, v) = (y .= x .+ v)

log!(M::VectorSpaceManifold, v, x, y) = (v .= y .- x)

function zero_tangent_vector!(M::VectorSpaceManifold, v, x)
    fill!(v, 0)
    return v
end

function project_point!(M::VectorSpaceManifold{<:TCoTSpaceType}, x)
    project_tangent!(M.M, x, M.point, x)
end

function project_tangent!(M::VectorSpaceManifold{<:TCoTSpaceType}, w, x, v)
    project_tangent!(M.M, w, M.point, v)
end


"""
    VectorBundle(M::Manifold, type::VectorSpaceType)

Vector bundle on manifold `M` of type `type`.
"""
struct VectorBundle{TVS<:VectorSpaceType, TM<:Manifold} <: Manifold
    type::TVS
    M::TM
end

function representation_size(M::VectorBundle)
    len_manifold = prod(representation_size(M.M))
    len_vs = prod(representation_size(M.M, M.type))
    return (len_manifold + len_vs,)
end

TangentBundle(M::Manifold) = VectorBundle(TangentSpace, M)
CotangentBundle(M::Manifold) = VectorBundle(CotangentSpace, M)

"""
    bundle_projection(M::VectorBundle, x::ProductRepr)

Bundle projection of point `x` from the vector bundle `M`.
"""
function bundle_projection(M::VectorBundle, x::ProductRepr)
    return x.parts[1]
end

function bundle_projection(M::VectorBundle, x::ProductArray)
    return x.parts[1]
end

function isapprox(M::VectorBundle, x, y; kwargs...)
    return isapprox(M.M, x.parts[1], y.parts[1]; kwargs...) && isapprox(x.parts[2], y.parts[2]; kwargs...)
end

function isapprox(M::VectorBundle, x, v, w; kwargs...)
    return isapprox(v.parts[1], w.parts[1]; kwargs...) && isapprox(v.parts[2], w.parts[2]; kwargs...)
end

manifold_dimension(M::VectorBundle) = manifold_dimension(M.M) + manifold_dimension(M.M, M.type)

function inner(M::VectorBundle, x, v, w)
    vsm = VectorSpaceManifold(M.type, M.M, x.parts[1])

    return inner(M.M, x.parts[1], v.parts[1], w.parts[1]) +
           inner(vsm, x.parts[2], v.parts[2], w.parts[2])
end

function distance(M::VectorBundle, x, y)
    vsm = VectorSpaceManifold(M.type, M.M, x.parts[1])

    dist_man = distance(M.M, x.parts[1], y.parts[1])
    dist_vec = distance(vsm, x.parts[2], y.parts[2])

    return sqrt(dist_man^2 + dist_vec^2)
end

function exp!(M::VectorBundle, y, x, v)
    vsm = VectorSpaceManifold(M.type, M.M, x.parts[1])

    exp!(M.M, y.parts[1], x.parts[1], v.parts[1])
    exp!(vsm, y.parts[2], x.parts[2], v.parts[2])
    return y
end

function log!(M::VectorBundle, v, x, y)
    vsm = VectorSpaceManifold(M.type, M.M, x.parts[1])

    log!(M.M, v.parts[1], x.parts[1], y.parts[1])
    log!(vsm, v.parts[2], x.parts[2], y.parts[2])
    return v
end

function zero_tangent_vector!(M::VectorBundle, v, x)
    vsm = VectorSpaceManifold(M.type, M.M, x.parts[1])

    zero_tangent_vector!(M.M, v.parts[1], x.parts[1])
    zero_tangent_vector!(vsm, v.parts[2], x.parts[2])
    return v
end

function project_point!(M::VectorBundle, x)
    project_point!(M.M, x.parts[1])
    project_tangent!(M.M, x.parts[2], x.parts[1], x.parts[2])
    return x
end

function project_tangent!(M::VectorBundle, w, x, v)
    project_tangent!(M.M, w.parts[1], x.parts[1], v.parts[1])
    project_tangent!(M.M, w.parts[2], x.parts[2], v.parts[2])
    return w
end
