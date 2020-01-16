
using LightGraphs
import LightGraphs: AbstractSimpleGraph, is_directed
import SimpleWeightedGraphs: AbstractSimpleWeightedGraph

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
    GraphManifold{G, M, T} <: AbstractPowerManifold{M}

Build a manifold, that is a [`PowerManifold`](@ref) of the [`Manifold`](@ref) `M` either on the edges or vertices
of a graph `G` depending on the [`GraphManifoldType`](@ref) `T`.

# Fields
* `G` is an `AbstractSimpleGraph`
* `M` is a [`Manifold`](@ref)
"""
struct GraphManifold{G<:AbstractSimpleGraph, TM, T<:GraphManifoldType} <: AbstractPowerManifold{TM}
    graph::G
    manifold::TM
end
GraphManifold(g::G,M::TM,::VertexManifold) where {G<:AbstractSimpleGraph, TM <: Manifold} = GraphManifold{G,TM,VertexManifold}(g,M)
GraphManifold(g::G,M::TM,::EdgeManifold) where {G<:AbstractSimpleGraph, TM <: Manifold} = GraphManifold{G,TM,EdgeManifold}(g,M)

@doc doc"""
    check_manifold_point(M::GraphManifold,x)

check whether `x` is a valid point on the [`GraphManifold`](@ref), i.e. its
length equals the number of vertices (for [`VertexManifold`](@ref)s) or
the number of edges (for [`EdgeManifold`](@ref)s) and that each element of `x`
passes the [`check_manifold_point`](@ref) test for the base manifold `M.manifold`.
"""
check_manifold_point(::GraphManifold,::Any...)
function check_manifold_point(
    M::GraphManifold{G,TM,VertexManifold},
    x;
    kwargs...) where {G <: AbstractSimpleGraph, TM <: Manifold}
    if size(x)[end] != nv(M.graph)
        return DomainError(length(x), "The number of elements in `x` ($(length(x)) does not match the number of nodes in the graph ($(nv(M.graph))).")
    end
    return check_manifold_point(PowerManifold(M.manifold,nv(M.graph)), x; kwargs...)
end
function check_manifold_point(
    M::GraphManifold{G,TM,EdgeManifold},
    x;
    kwargs...) where {G <: AbstractSimpleGraph,TM<:Manifold}
    if size(x)[end] != ne(M.graph)
        return DomainError(length(x), "The number of elements in `x` ($(length(x)) does not match the number of edges in the graph ($(ne(M.graph))).")
    end
    return check_manifold_point(PowerManifold(M.manifold,ne(M.graph)), x; kwargs...)
end

@doc doc"""
    check_tangent_vector(M::GraphManifold,x,v)

check whether `x` is a valid point on the [`GraphManifold`](@ref), and
`v` it from its tangent space, i.e. its
length equals the number of vertices (for [`VertexManifold`](@ref)s) or
the number of edges (for [`EdgeManifold`](@ref)s) and that each element of `v`
together with its corresponding einty of `x` passes the
[`check_tangent_vector`](@ref) test for the base manifold `M.manifold`.
"""
check_tangent_vector(::GraphManifold,::Any...)
function check_tangent_vector(
    M::GraphManifold{G, TM, VertexManifold},
    x,
    v;
    kwargs...) where {G <: AbstractSimpleGraph, TM <: Manifold}
    if size(x)[end] != nv(M.graph)
        return DomainError(length(x), "The number of elements in `x` ($(length(x)) does not match the number of nodes in the graph ($(nv(M.graph))).")
    end
    if size(v)[end] != nv(M.graph)
        return DomainError(length(v), "The number of elements in `v` ($(length(v)) does not match the number of nodes in the graph ($(nv(M.graph))).")
    end
    return check_tangent_vector(PowerManifold(M.manifold,nv(M.graph)), x, v; kwargs...)
end
function check_tangent_vector(
    M::GraphManifold{G, TM, EdgeManifold},
    x,
    v;
    kwargs...) where {G <: AbstractSimpleGraph, TM <: Manifold}
    if size(x)[end] != ne(M.graph)
        return DomainError(length(x), "The number of elements in `x` ($(length(x)) does not match the number of edges in the graph ($(ne(M.graph))).")
    end
    if size(v)[end] != ne(M.graph)
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
    incident_log(M::GraphManifold, x)

return the tangent vector on the (vertex) [`GraphManifold`](@ref), where at
each node the sum of the [`log`](@ref)s to incident nodes is computed.
For a `SimpleGraph`, an egde is interpreted as double edge in the corresponding
SimpleDiGraph

If the internal graph is a `SimpleWeightedGraph` the weighted sum of the
tangent vectors is computed.
"""
function incident_log(M::GraphManifold{G, TM, VertexManifold}, x) where {G <: AbstractSimpleGraph, TM<:Manifold}
    v = zero_tangent_vector(M,x)
    return incident_log!(M, v, x)
end
function incident_log!(M::GraphManifold{G, TM, VertexManifold}, v, x) where {G <: AbstractSimpleGraph, TM<:Manifold}
    rep_size = representation_size(M.manifold)
    rsC = rep_size_to_colons(rep_size)
    for e in edges(M.graph)
        vw = _write(rep_size, v, src(e))
        v[rsC..., src(e)] = log(M.manifold, _read(rep_size, x, src(e)), _read(rep_size, x, dst(e)) )
        if !is_directed(M.graph)
            v[rsC...,dst(e)] += log(M.manifold, _read(rep_size, x, dst(e)), _read(rep_size, x, dst(e)) )
        end
    end
    return v
end
function incident_log!(M::GraphManifold{G, TM, VertexManifold}, v, x) where {G <: AbstractSimpleWeightedGraph, TM<:Manifold}
    rep_size = representation_size(M.manifold)
    rsC = rep_size_to_colons(rep_size)
    for e in edges(M.graph)
        v[rsC...,src(e)] += M.graph.weight[src(e),dst(e)]*log(M.manifold, _read(rep_size, x, src(e)), _read(rep_size, x, dst(e)))
        if !is_directed(M.graph)
            v[rsC...,dst(e)] += M.graph.weight[dst(e),src(e)]*log(M.manifold, _read(rep_size, x, dst(e)), _read(rep_size, x, src(e)))
        end
    end
    return v
end

@doc doc"""
    manifold_dimension(N::GraphManifold{G,M,VertexManifold})

returns the manifold dimension of the [`GraphManifold`](@ref) `N` on the vertices of
a graph $G=(V,E)$, i.e.
````math
d_{\mathcal N} = \lvert V \rVert d_{\mathcal M}.
````
"""
function manifold_dimension(M::GraphManifold{<: AbstractSimpleGraph,<: Manifold, VertexManifold})
    return manifold_dimension(M.manifold)*nv(M.graph)
end
@doc doc"""
    manifold_dimension(N::GraphManifold{G,M,EdgeManifold})

returns the manifold dimension of the [`GraphManifold`](@ref) `N` on the edges of
a graph $G=(V,E)$, i.e.
````math
d_{\mathcal N} = \lvert E \rVert d_{\mathcal M}.
````
"""
function manifold_dimension(M::GraphManifold{<: AbstractSimpleGraph, <: Manifold, EdgeManifold})
    return manifold_dimension(M.manifold)*ne(M.graph)
end

@doc doc"""
    representationsize(N::GraphManifold{G,M,VertexManifold})

returns the representation size of a point on the [`GraphManifold`](@ref) on the vertices of $G=(V,E),
which is the representation size of `M` with $\lvert V \rvert$ added.
"""
function representation_size(M::GraphManifold{<: AbstractSimpleGraph,<: Manifold, VertexManifold})
    return (representation_size(M.manifold)..., nv(M.graph))
end
@doc doc"""
    representationsize(N::GraphManifold{G,M,EdgeManifold})

returns the representation size of a point on the [`GraphManifold`](@ref) on the vertices of $G=(V,E),
which is the representation size of `M` with $\lvert E \rvert$ added.
"""
function representation_size(M::GraphManifold{<: AbstractSimpleGraph, <: Manifold, EdgeManifold})
    return (representation_size(M.manifold)..., ne(M.graph))
end

function zero_tangent_vector(M::GraphManifold{G,TM,VertexManifold}, x) where {G <: AbstractSimpleGraph, TM<:Manifold}
    return zero_tangent_vector(PowerManifold(M.manifold,nv(M.graph)), x)
end
function zero_tangent_vector(M::GraphManifold{G,TM,EdgeManifold}, x) where {G <: AbstractSimpleGraph, TM<:Manifold}
    return zero_tangent_vector(PowerManifold(M.manifold,ne(M.graph)), x)
end
