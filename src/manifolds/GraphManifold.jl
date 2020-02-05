@doc raw"""
    GraphManifoldType

This type represents the type of data on the graph that the [`GraphManifold`](@ref)
represents.
"""
abstract type GraphManifoldType end

@doc raw"""
    EdgeManifoldManifold <: GraphManifoldType

A type for a [`GraphManifold`](@ref) where the data is given on the edges.
"""
struct EdgeManifold <: GraphManifoldType end

@doc raw"""
    VectexGraphManifold <: GraphManifoldType

A type for a [`GraphManifold`](@ref) where the data is given on the vertices.
"""
struct VertexManifold <: GraphManifoldType end

@doc raw"""
    GraphManifold{G, M, T} <: AbstractPowerManifold{M,NestedPowerRepresentation}

Build a manifold, that is a [`PowerManifold`](@ref) of the [`Manifold`](@ref) `M` either on the edges or vertices
of a graph `G` depending on the [`GraphManifoldType`](@ref) `T`.

# Fields
* `G` is an `AbstractSimpleGraph`
* `M` is a [`Manifold`](@ref)
"""
struct GraphManifold{G<:AbstractGraph,TM,T<:GraphManifoldType} <:
       AbstractPowerManifold{TM,NestedPowerRepresentation}
    graph::G
    manifold::TM
end

function GraphManifold(g::G, M::TM, ::VertexManifold) where {G<:AbstractGraph,TM<:Manifold}
    return GraphManifold{G,TM,VertexManifold}(g, M)
end
function GraphManifold(g::G, M::TM, ::EdgeManifold) where {G<:AbstractGraph,TM<:Manifold}
    return GraphManifold{G,TM,EdgeManifold}(g, M)
end

const EdgeGraphManifold = GraphManifold{<:AbstractGraph,<:Manifold,EdgeManifold}
const VertexGraphManifold = GraphManifold{<:AbstractGraph,<:Manifold,VertexManifold}

@doc raw"""
    check_manifold_point(M::GraphManifold, p)

Check whether `p` is a valid point on the [`GraphManifold`](@ref), i.e. its length equals the number of vertices
(for [`VertexManifold`](@ref)s) or the number of edges (for [`EdgeManifold`](@ref)s) and that each element of `p`
passes the [`check_manifold_point`](@ref) test for the base manifold `M.manifold`.
"""
check_manifold_point(::GraphManifold, ::Any...)
function check_manifold_point(M::VertexGraphManifold, p; kwargs...)
    if size(p) != (nv(M.graph),)
        return DomainError(
            length(p),
            "The number of points in `x` ($(length(p))) does not match the number of nodes in the graph ($(nv(M.graph))).",
        )
    end
    PM = PowerManifold(M.manifold, NestedPowerRepresentation(), nv(M.graph))
    return check_manifold_point(PM, p; kwargs...)
end
function check_manifold_point(M::EdgeGraphManifold, p; kwargs...)
    if size(p) != (ne(M.graph),)
        return DomainError(
            length(p),
            "The number of points in `x` ($(size(p))) does not match the number of edges in the graph ($(ne(M.graph))).",
        )
    end
    PM = PowerManifold(M.manifold, NestedPowerRepresentation(), ne(M.graph))
    return check_manifold_point(PM, p; kwargs...)
end

@doc raw"""
    check_tangent_vector(M::GraphManifold, p, X)

Check whether `p` is a valid point on the [`GraphManifold`](@ref), and
`X` it from its tangent space, i.e. its
length equals the number of vertices (for [`VertexManifold`](@ref)s) or
the number of edges (for [`EdgeManifold`](@ref)s) and that each element of `X`
together with its corresponding entry of `p` passes the
[`check_tangent_vector`](@ref) test for the base manifold `M.manifold`.
"""
check_tangent_vector(::GraphManifold, ::Any...)
function check_tangent_vector(M::VertexGraphManifold, p, X; kwargs...)
    if size(p) != (nv(M.graph),)
        return DomainError(
            length(p),
            "The number of points in `x` ($(size(p)) does not match the number of nodes in the graph ($(nv(M.graph))).",
        )
    end
    if size(X) != (nv(M.graph),)
        return DomainError(
            length(X),
            "The number of points in `v` ($(size(X)) does not match the number of nodes in the graph ($(nv(M.graph))).",
        )
    end
    PM = PowerManifold(M.manifold, NestedPowerRepresentation(), nv(M.graph))
    return check_tangent_vector(PM, p, X; kwargs...)
end
function check_tangent_vector(M::EdgeGraphManifold, p, X; kwargs...)
    if size(p) != (ne(M.graph),)
        return DomainError(
            length(p),
            "The number of elements in `x` ($(size(p)) does not match the number of edges in the graph ($(ne(M.graph))).",
        )
    end
    if size(X) != (ne(M.graph),)
        return DomainError(
            length(X),
            "The number of elements in `v` ($(size(X)) does not match the number of edges in the graph ($(ne(M.graph))).",
        )
    end
    PM = PowerManifold(M.manifold, NestedPowerRepresentation(), ne(M.graph))
    return check_tangent_vector(PM, p, X; kwargs...)
end

get_iterator(M::EdgeGraphManifold) = 1:ne(M.graph)
get_iterator(M::VertexGraphManifold) = 1:nv(M.graph)

@doc raw"""
    incident_log(M::GraphManifold, x)

Return the tangent vector on the (vertex) [`GraphManifold`](@ref), where at
each node the sum of the [`log`](@ref)s to incident nodes is computed.
For a `SimpleGraph`, an egde is interpreted as double edge in the corresponding
SimpleDiGraph

If the internal graph is a `SimpleWeightedGraph` the weighted sum of the
tangent vectors is computed.
"""
function incident_log(M::VertexGraphManifold, p)
    v = zero_tangent_vector(M, p)
    return incident_log!(M, v, p)
end

function incident_log!(M::VertexGraphManifold, X, p)
    rep_size = representation_size(M.manifold)
    for e in edges(M.graph)
        vw = _write(M, rep_size, X, src(e))
        X[src(e)] +=
            log(M.manifold, _read(M, rep_size, p, src(e)), _read(M, rep_size, p, dst(e)))
        if !is_directed(M.graph)
            X[dst(e)] += log(
                M.manifold,
                _read(M, rep_size, p, dst(e)),
                _read(M, rep_size, p, src(e)),
            )
        end
    end
    return X
end
function incident_log!(
    M::GraphManifold{<:AbstractSimpleWeightedGraph,<:Manifold,VertexManifold},
    X,
    p,
)
    rep_size = representation_size(M.manifold)
    for e in edges(M.graph)
        X[src(e)] += (
            get_weight(M.graph, src(e), dst(e)) * log(
                M.manifold,
                _read(M, rep_size, p, src(e)),
                _read(M, rep_size, p, dst(e)),
            )
        )
        if !is_directed(M.graph)
            X[dst(e)] += (
                get_weight(M.graph, dst(e), src(e)) * log(
                    M.manifold,
                    _read(M, rep_size, p, dst(e)),
                    _read(M, rep_size, p, src(e)),
                )
            )
        end
    end
    return X
end

@doc raw"""
    manifold_dimension(N::GraphManifold{G,M,VertexManifold})

returns the manifold dimension of the [`GraphManifold`](@ref) `N` on the vertices of
a graph $G=(V,E)$, i.e.
````math
\dim(\mathcal N) = \lvert V \rVert \dim(\mathcal M).
````
"""
function manifold_dimension(M::VertexGraphManifold)
    return manifold_dimension(M.manifold) * nv(M.graph)
end
@doc raw"""
    manifold_dimension(N::GraphManifold{G,M,EdgeManifold})

returns the manifold dimension of the [`GraphManifold`](@ref) `N` on the edges of
a graph $G=(V,E)$, i.e.
````math
\dim(\mathcal N) = \lvert E \rVert \dim(\mathcal M).
````
"""
function manifold_dimension(M::EdgeGraphManifold)
    return manifold_dimension(M.manifold) * ne(M.graph)
end

function _show_graph_manifold(io::IO, M; man_desc = "", pre = "")
    println(io, "GraphManifold\nGraph:")
    sg = sprint(show, "text/plain", M.graph, context = io, sizehint = 0)
    sg = replace(sg, '\n' => "\n$(pre)")
    println(io, pre, sg)
    println(io, "Manifold$(man_desc):")
    sm = sprint(show, "text/plain", M.manifold, context = io, sizehint = 0)
    sm = replace(sm, '\n' => "\n$(pre)")
    print(io, pre, sm)
    return nothing
end

function show(io::IO, mime::MIME"text/plain", M::EdgeGraphManifold)
    _show_graph_manifold(io, M; man_desc = " on edges", pre = " ")
end
function show(io::IO, mime::MIME"text/plain", M::VertexGraphManifold)
    _show_graph_manifold(io, M; man_desc = " on vertices", pre = " ")
end
