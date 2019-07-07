
abstract type AbstractVectorSpace end

"""
    VectorSpaceType

Abstract type for tangent space, cotangent space, their tensor products,
exterior products etc.
"""
abstract type VectorSpaceType end

struct TangentSpaceType <: VectorSpaceType end
struct CotangentSpaceType <: VectorSpaceType end

"""
    VectorSpace(M, x, type)

Vector space of type `type` at point `x` from manifold `M`.
"""
struct VectorSpace{TType<:VectorSpaceType, TM<:Manifold, TP} <: AbstractVectorSpace
    type::TType
    M::TM
    point::TP
end

TangentSpace(M::Manifold, point) = VectorSpace(TangentSpaceType(), M, point)
CotangentSpace(M::Manifold, point) = VectorSpace(CotangentSpaceType(), M, point)
TCoTSpaceType = Union{TangentSpaceType, CotangentSpaceType}

"""
    project_vector!(vs::VectorSpace, v, w)

Project vector `w` from the vector space `vs` and save the result to `v`.
"""
function project_vector!(vs::VectorSpace, v, w)
    error("project_vector! not implemented for vector space $(typeof(vs)) and vectors of type $(typeof(v)) and $(typeof(w)).")
end

function project_vector!(vs::VectorSpace{<:TCoTSpaceType}, v, w)
    project_tangent!(vs.M, v, vs.point, w)
    return  v
end

function similar_result(M::VectorSpace, f, x1::AbstractArray, x...)
    return similar_result(M.M, f, x1, x...)
end

"""
    manifold_dimension(M::Manifold, t::VectorSpaceType)

Dimension of the vector space of type `t` over manifold `M`.
"""
function manifold_dimension(M::Manifold, t::VectorSpaceType)
    error("manifold_dimension not implemented for manifold $(typeof(M)) and vector space of type $(typeof(t)).")
end

manifold_dimension(M::Manifold, ::TCoTSpaceType) = manifold_dimension(M)

"""
    VectorSpaceManifold(vs::AbstractVectorSpace)

Manifold corresponding to the vector space `vs`.
"""
struct VectorSpaceManifold{TVS <: AbstractVectorSpace} <: Manifold
    vs::TVS
end


manifold_dimension(M::VectorSpaceManifold{<:VectorSpace{<:TCoTSpaceType}}) = manifold_dimension(M.vs.M)

function representation_size(M::VectorSpaceManifold{<:VectorSpace{<:TCoTSpaceType}}, ::Type{T}) where {T}
    representation_size(M.vs.M, T)
end

@inline inner(::VectorSpaceManifold{<:VectorSpace{<:TCoTSpaceType}}, x, v, w) = dot(v, w)

distance(::VectorSpaceManifold{<:VectorSpace}, x, y) = norm(x-y)
norm(::VectorSpaceManifold{<:VectorSpace{TCoTSpaceType}}, x, v) = norm(v)

exp!(M::VectorSpaceManifold{<:VectorSpace}, y, x, v) = (y .= x + v)

log!(M::VectorSpaceManifold{<:VectorSpace}, v, x, y) = (v .= y - x)

function zero_tangent_vector!(M::VectorSpaceManifold{<:VectorSpace}, v, x)
    fill!(v, 0)
    return v
end

function project_point!(M::VectorSpaceManifold{<:VectorSpace{TCoTSpaceType}}, x)
    error("TODO")
end

function project_tangent!(M::VectorSpaceManifold{<:VectorSpace{TCoTSpaceType}}, w, x, v)
    error("TODO")
end


"""
    VectorBundle(M::Manifold, type::VectorSpaceType)

Vector bundle on manifold `M` of type `type.`
"""
struct VectorBundle{TVS<:VectorSpaceType, TM<:Manifold} <: Manifold
    type::TVS
    M::TM
end

TangentBundle(M::Manifold) = VectorBundle(TangentSpaceType(), M)
CotangentBundle(M::Manifold) = VectorBundle(CotangentSpaceType(), M)

"""
    VectorBundleRepr(x, v)

A representation of vector bundle-based values (points, tangent vectors,
cotangent vectors).
Point on the underlying manifold relates to `x` and vector from
the vector space relates to `v`.
"""
struct VectorBundleRepr{TX, TV}
    x::TX
    v::TV
end

(+)(v1::VectorBundleRepr, v2::VectorBundleRepr) = VectorBundleRepr(v1.x + v2.x, v1.v + v2.v)
(-)(v1::VectorBundleRepr, v2::VectorBundleRepr) = VectorBundleRepr(v1.x - v2.x, v1.v - v2.v)
(-)(v::VectorBundleRepr) = VectorBundleRepr(-v.x, -v.v)
(*)(a::Number, v::VectorBundleRepr) = VectorBundleRepr(a*v.x, a*v.v)


function similar_result(M::VectorBundle, f, x1::VectorBundleRepr, x...)
    vs = VectorSpace(M.type, M.M, x1.x)
    return VectorBundleRepr(similar_result(M.M, f, x1.x), similar_result(vs, f, x1.v))
end

function isapprox(M::VectorBundle, x, y; kwargs...)
    return isapprox(M.M, x.x, y.x; kwargs...) && isapprox(x.v, y.v; kwargs...)
end

function isapprox(M::VectorBundle, x, v, w; kwargs...)
    return isapprox(v.x, w.x; kwargs...) && isapprox(v.v, w.v; kwargs...)
end

manifold_dimension(M::VectorBundle) = manifold_dimension(M.M) + manifold_dimension(M.M, M.type)

function inner(::VectorBundle, x, v, w)
    vsm = VectorSpaceManifold(VectorSpace(M.type, M.M, x.x))

    return inner(M.M, x.x, v.x, w.x) + inner(vsm, x.v, v.v, w.v)
end

function distance(M::VectorBundle, x, y)
    vsm = VectorSpaceManifold(VectorSpace(M.type, M.M, x.x))

    dist_man = distance(M.M, x.x, y.x)
    dist_vec = distance(vsm, x.v, y.v)

    return sqrt(dist_man^2 + dist_vec^2)
end

function exp!(M::VectorBundle, y, x, v)
    vsm = VectorSpaceManifold(VectorSpace(M.type, M.M, x.x))

    exp!(M.M, y.x, x.x, v.x)
    exp!(vsm, y.v, x.v, v.v)
    return y
end

function log!(M::VectorBundle, v, x, y)
    vsm = VectorSpaceManifold(VectorSpace(M.type, M.M, x.x))

    log!(M.M, v.x, x.x, y.x)
    log!(vsm, v.v, x.v, y.v)
    return v
end

function zero_tangent_vector!(M::VectorBundle, v, x)
    fill!(v.x, 0)
    fill!(v.v, 0)
    return v
end

function project_point!(M::VectorBundle, x)
    project_point!(M.M, x.x)
    project_tangent!(M.M, x.v, x.x, x.v)
    return x
end

function project_tangent!(M::VectorBundle, w, x, v)
    project_tangent!(M.M, w.x, x.x, v.x)
    project_tangent!(M.M, w.v, x.x, v.v)
    return w
end
