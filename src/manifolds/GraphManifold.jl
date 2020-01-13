
using LightGraphs
import LightGraphs: AbstractSimpleGraph

@doc doc"""
    GraphManifoldType

This type represents the type of data on the graph that the [`GraphManifold`](@ref)
represents.
"""
abstract type GraphManifoldType end

@doc doc"""
    EdgeManifoldManifold <: GraphManifoldType

A type for a [`GraphManifold`](@ref) where the data is given on the edges.
"""
struct EdgeManifold <: GraphManifoldType end

@doc doc"""
    VectexGraphManifold <: GraphManifoldType

A type for a [`GraphManifold`](@ref) where the data is given on the vertices.
"""
struct VertexManifold <: GraphManifoldType end

@doc doc"""
    GraphManifold{G, M, T} <: AbstractPowerManifold

Build a manifold, that is a [`PowerManifold`](@ref) either on the
edges or vertices of a graph `G` depending on the [`GraphManifoldType`](@ref) `T`.


# Fields
* `G` is an `AbstractSimpleGraph`
* `M` is a [`Manifold`](@ref)
* `T` is a [`GraphManifoldType`](@ref), i.e. whether data is given on vertices
or edges.
"""
struct GraphManifold{G<:AbstractSimpleGraph, TM, T<:GraphManifoldType} <: AbstractPowerManifold{TM}
    graph::G
    manifold::TM
end
GraphManifold(g::G,M::TM,::VertexManifold) where {G<:AbstractSimpleGraph, TM <: Manifold} = GraphManifold{G,TM,VertexManifold}(g,M)
GraphManifold(g::G,M::TM,::EdgeManifold) where {G<:AbstractSimpleGraph, TM <: Manifold} = GraphManifold{G,TM,EdgeManifold}(g,M)
@doc doc"""
    check_manifold_point(M::GraphManifold,x)

check whether `x` is a valid point on the [`GraphManifold`](@ref)
"""
check_manifold_point(::GraphManifold,::Any)
function check_manifold_point(
    M::GraphManifold{G,TM,VertexManifold},
    x;
    kwargs...) where {G <: AbstractSimpleGraph, TM <: Manifold}
    if length(x) != nv(M.graph)
        return DomainError(length(x), "The number of elements in `x` ($(length(x)) does not match the number of nodes in the graph ($(nv(M.graph))).")
    end
    return check_manifold_point(PowerManifold(M.manifold,nv(M.graph)), x; kwargs...)
end
function check_manifold_point(
    M::GraphManifold{G,TM,EdgeManifold},
    x;
    kwargs...) where {G <: AbstractSimpleGraph,TM<:Manifold}
    if length(x) != ne(M.graph)
        return DomainError(length(x), "The number of elements in `x` ($(length(x)) does not match the number of edges in the graph ($(ne(M.graph))).")
    end
    return check_manifold_point(PowerManifold(M.manifold,nv(M.graph)), x; kwargs...)
end

function check_tangent_vector(
    M::GraphManifold{G, TM, VertexManifold},
    x,
    v;
    kwargs...) where {G <: AbstractSimpleGraph, TM <: Manifold}
    if length(x) != nv(M.graph)
        return DomainError(length(x), "The number of elements in `x` ($(length(x)) does not match the number of nodes in the graph ($(nv(M.graph))).")
    end
    if length(v) != nv(M.graph)
        return DomainError(length(v), "The number of elements in `v` ($(length(v)) does not match the number of nodes in the graph ($(nv(M.graph))).")
    end
    return check_tangent_vector(PowerManifold(M.manifold,nv(M.graph)), x, v; kwargs...)
end
function check_tangent_vector(
    M::GraphManifold{G, TM, EdgeManifold},
    x,
    v;
    kwargs...) where {G <: AbstractSimpleGraph, TM <: Manifold}
    if length(x) != ne(M.graph)
        return DomainError(length(x), "The number of elements in `x` ($(length(x)) does not match the number of edges in the graph ($(ne(M.graph))).")
    end
    if length(v) != ne(M.graph)
        return DomainError(length(v), "The number of elements in `v` ($(length(v)) does not match the number of edges in the graph ($(ne(M.graph))).")
    end
    return check_tangent_vector(PowerManifold(M.manifold,ne(M.graph)), x, v; kwargs...)
end


function get_iterator(M::GraphManifold{<: AbstractSimpleGraph, <:Manifold, EdgeManifold})
    return 1:ne(M.graph)
end
function get_iterator(M::GraphManifold{<: AbstractSimpleGraph, <:Manifold, VertexManifold})
    return 1:nv(M.graph)
end

@doc doc"""
    incident_log(M, x)

return the tangent vector on the (vertex) [`GraphManifold`](@ref), where at
each node the sum of the [`log`](@ref)s to incident nodes is summed.
"""
function incident_log(M::GraphManifold{G, TM, VertexManifold}, x) where {G <: AbstractSimpleGraph, TM<:Manifold}
    v = zero_tangent_vector(M,x)
    return incident_log!(M, v, x)
end
function incident_log!(M::GraphManifold{G, TM, VertexManifold}, v, x) where {G <: AbstractSimpleGraph, TM<:Manifold}
    for e in edges(M.graph)
        v[src(e)] += log(M.manifold, x[src(e)], x[dst(e)])
    end
    return v;
end

function manifold_dimension(M::GraphManifold{<: AbstractSimpleGraph,<: Manifold, VertexManifold})
    return manifold_dimension(M.manifold)*nv(M.graph)
end
function manifold_dimension(M::GraphManifold{<: AbstractSimpleGraph, <: Manifold, EdgeManifold})
    return manifold_dimension(M.manifold)*ne(M.graph)
end

function representation_size(M::GraphManifold{<: AbstractSimpleGraph,<: Manifold, VertexManifold})
    return (representation_size(M.manifold)..., nv(M.graph))
end
function representation_size(M::GraphManifold{<: AbstractSimpleGraph, <: Manifold, EdgeManifold})
    return (representation_size(M.manifold)..., ne(M.graph))
end
