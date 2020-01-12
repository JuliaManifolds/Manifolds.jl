
using LightGraphs
import LightGraphs: AbstractSimpleGraph

@doc doc"""
    GraphManifoldType

This type represents the type of data on the graph that the [`GraphManifold`](@ref)
represents.
"""
abstract type GraphManifoldType end

@doc doc"""
    EdgeGraphManifold <: GraphManifoldType

A type for a [`GraphManifold`](@ref) where the data is given on the edges.
"""
struct EdgeGraph <: GraphManifoldType end

@doc doc"""
    VectexGraphManifold <: GraphManifoldType

A type for a [`GraphManifold`](@ref) where the data is given on the vertices.
"""
struct VertexGraph <: GraphManifoldType end

@doc doc"""
    GraphManifold{G,M,T}

Build a manifold, that is basically a [`PowerManifold`](@ref) either on the
edges or vertices of a graph.

# Fields
* `G` is an `AbstractSimpleGraph`
* `M` is a [`Manifold`](@ref)
* `T` is a [`GraphManifoldType`](@ref), i.e. whether data is given on vertices
or edges.
"""
struct GraphManifold{
        G<:AbstractSimpleGraph,
        M<:Manifold,
        T<:GraphManifoldType
    } <: Manifold
    graph::G
    manifold::M
    type::T
end

@doc doc"""
    check_manifold_point(M::GraphManifold,x)

check whether `x` is a valid point on the [`GraphManifold`](@ref)
"""
check_manifold_point(::GraphManifold,::Any)

function check_manifold_point(
    M::GraphManifold{G,TM,VertexGraph},
    x;
    kwargs...) where {G <: AbstractSimpleGraph,TM<:Manifold}
    for i in nv(M.graph)
        imp = check_manifold_point(M.manifold, x[i]; kwargs...)
        imp === nothing || return imp
    end
end
function check_manifold_point(
    M::GraphManifold{G,TM,EdgeGraph},
    x;
    kwargs...) where {G <: AbstractSimpleGraph,TM<:Manifold}
    for e in edges(M.graph)
        imp = check_manifold_point(M.manifold, x[e]; kwargs...)
        imp === nothing || return imp
    end
end

@doc doc"""
    incident_log(M, x)

return the tangent vector on the (vertex) [`GraphManifold`](@ref), where at
each node the sum of the [`log`](@ref)s to incident nodes is summed.
"""
function incident_log(M::GraphManifold{G,TM,VertexGraph}, x) where {G <: AbstractSimpleGraph,TM<:Manifold}
    v = zero_tangent_vector(M,x)
    return incident_log!(M, v, x)
end
function incident_log!(M::GraphManifold{G,TM,VertexGraph}, v, x) where {G <: AbstractSimpleGraph,TM<:Manifold}
    for e in edges(M.graph)
        v[src(e)] += log(M.manifold, x[src(e)], x[dst(e)])
    end
end