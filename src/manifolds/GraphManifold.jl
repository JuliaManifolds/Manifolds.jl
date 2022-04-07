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
    GraphManifold{G,𝔽,M,T} <: AbstractPowerManifold{𝔽,M,NestedPowerRepresentation}

Build a manifold, that is a [`PowerManifold`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/manifolds.html#ManifoldsBase.PowerManifold) of the [`AbstractManifold`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/types.html#ManifoldsBase.AbstractManifold)  `M` either on
the edges or vertices of a graph `G` depending on the [`GraphManifoldType`](@ref) `T`.

# Fields
* `G` is an `AbstractSimpleGraph`
* `M` is a `AbstractManifold`
"""
struct GraphManifold{G<:AbstractGraph,𝔽,TM,T<:GraphManifoldType} <:
       AbstractPowerManifold{𝔽,TM,NestedPowerRepresentation}
    graph::G
    manifold::TM
end

function GraphManifold(
    g::G,
    M::TM,
    ::VertexManifold,
) where {G<:AbstractGraph,𝔽,TM<:AbstractManifold{<:𝔽}}
    return GraphManifold{G,𝔽,TM,VertexManifold}(g, M)
end
function GraphManifold(
    g::G,
    M::TM,
    ::EdgeManifold,
) where {G<:AbstractGraph,𝔽,TM<:AbstractManifold{<:𝔽}}
    return GraphManifold{G,𝔽,TM,EdgeManifold}(g, M)
end

const EdgeGraphManifold{𝔽} =
    GraphManifold{<:AbstractGraph,𝔽,<:AbstractManifold{𝔽},EdgeManifold}
const VertexGraphManifold{𝔽} =
    GraphManifold{<:AbstractGraph,𝔽,<:AbstractManifold{𝔽},VertexManifold}

@doc raw"""
    check_point(M::GraphManifold, p)

Check whether `p` is a valid point on the [`GraphManifold`](@ref), i.e. its length equals the number of vertices
(for [`VertexManifold`](@ref)s) or the number of edges (for [`EdgeManifold`](@ref)s) and that each element of `p`
passes the [`check_point`](@ref) test for the base manifold `M.manifold`.
"""
check_point(::GraphManifold, ::Any...)
function check_point(M::VertexGraphManifold, p; kwargs...)
    PM = PowerManifold(M.manifold, NestedPowerRepresentation(), nv(M.graph))
    return check_point(PM, p; kwargs...)
end
function check_point(M::EdgeGraphManifold, p; kwargs...)
    PM = PowerManifold(M.manifold, NestedPowerRepresentation(), ne(M.graph))
    return check_point(PM, p; kwargs...)
end

@doc raw"""
    check_vector(M::GraphManifold, p, X; kwargs...)

Check whether `p` is a valid point on the [`GraphManifold`](@ref), and
`X` it from its tangent space, i.e. its
length equals the number of vertices (for [`VertexManifold`](@ref)s) or
the number of edges (for [`EdgeManifold`](@ref)s) and that each element of `X`
together with its corresponding entry of `p` passes the
[`check_vector`](@ref) test for the base manifold `M.manifold`.
"""
check_vector(::GraphManifold, ::Any...)
function check_vector(M::VertexGraphManifold, p, X; kwargs...)
    PM = PowerManifold(M.manifold, NestedPowerRepresentation(), nv(M.graph))
    return check_vector(PM, p, X; kwargs...)
end
function check_vector(M::EdgeGraphManifold, p, X; kwargs...)
    PM = PowerManifold(M.manifold, NestedPowerRepresentation(), ne(M.graph))
    return check_vector(PM, p, X; kwargs...)
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
    v = zero_vector(M, p)
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
    M::GraphManifold{<:AbstractSimpleWeightedGraph,𝔽,<:AbstractManifold{𝔽},VertexManifold},
    X,
    p,
) where {𝔽}
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
    manifold_dimension(N::GraphManifold{G,𝔽,M,VertexManifold})

returns the manifold dimension of the [`GraphManifold`](@ref) `N` on the vertices of
a graph $G=(V,E)$, i.e.
````math
\dim(\mathcal N) = \lvert V \rvert \dim(\mathcal M),
````
where $\mathcal M$ is the manifold of the data on the nodes.
"""
function manifold_dimension(M::VertexGraphManifold)
    return manifold_dimension(M.manifold) * nv(M.graph)
end
@doc raw"""
    manifold_dimension(N::GraphManifold{G,𝔽,M,EdgeManifold})

returns the manifold dimension of the [`GraphManifold`](@ref) `N` on the edges of
a graph $G=(V,E)$, i.e.
````math
\dim(\mathcal N) = \lvert E \rvert \dim(\mathcal M),
````
where $\mathcal M$ is the manifold of the data on the edges.
"""
function manifold_dimension(M::EdgeGraphManifold)
    return manifold_dimension(M.manifold) * ne(M.graph)
end

power_dimensions(M::EdgeGraphManifold) = (ne(M.graph),)
power_dimensions(M::VertexGraphManifold) = (nv(M.graph),)

function _show_graph_manifold(io::IO, M; man_desc="", pre="")
    println(io, "GraphManifold\nGraph:")
    sg = sprint(show, "text/plain", M.graph, context=io, sizehint=0)
    sg = replace(sg, '\n' => "\n$(pre)")
    println(io, pre, sg)
    println(io, "AbstractManifold$(man_desc):")
    sm = sprint(show, "text/plain", M.manifold, context=io, sizehint=0)
    sm = replace(sm, '\n' => "\n$(pre)")
    print(io, pre, sm)
    return nothing
end

function Base.show(io::IO, mime::MIME"text/plain", M::EdgeGraphManifold)
    return _show_graph_manifold(io, M; man_desc=" on edges", pre=" ")
end
function Base.show(io::IO, mime::MIME"text/plain", M::VertexGraphManifold)
    return _show_graph_manifold(io, M; man_desc=" on vertices", pre=" ")
end
