@inline function allocate_result(M::FiberBundle, f::TF) where {TF}
    p = allocate_result(M.manifold, f)
    X = allocate_result(Fiber(M.manifold, p, M.type), f)
    return ArrayPartition(p, X)
end

"""
    getindex(p::ArrayPartition, M::FiberBundle, s::Symbol)
    p[M::FiberBundle, s]

Access the element(s) at index `s` of a point `p` on a [`FiberBundle`](@ref) `M` by
using the symbols `:point` and `:vector` or `:fiber` for the base and vector or fiber
component, respectively.
"""
@inline function Base.getindex(p::ArrayPartition, M::FiberBundle, s::Symbol)
    (s === :point) && return p.x[1]
    (s === :vector || s === :fiber) && return p.x[2]
    return throw(DomainError(s, "unknown component $s on $M."))
end

function get_vector(M::FiberBundle, p::ArrayPartition, c::AbstractVector, B::AbstractBasis)
    n = manifold_dimension(M.manifold)
    xp1, xp2 = submanifold_components(M, p)
    F = Fiber(M.manifold, xp1, M.type)
    return ArrayPartition(
        get_vector(M.manifold, xp1, c[1:n], B),
        get_vector(F, xp2, c[(n + 1):end], B),
    )
end
function get_vector(
        M::FiberBundle, p::ArrayPartition, c::AbstractVector,
        B::CachedBasis{𝔽, <:AbstractBasis{𝔽}, <:FiberBundleBasisData},
    ) where {𝔽}
    n = manifold_dimension(M.manifold)
    xp1, xp2 = submanifold_components(M, p)
    F = Fiber(M.manifold, xp1, M.type)
    return ArrayPartition(
        get_vector(M.manifold, xp1, c[1:n], B.data.base_basis),
        get_vector(F, xp2, c[(n + 1):end], B.data.fiber_basis),
    )
end

function get_vectors(
        M::FiberBundle, p::ArrayPartition,
        B::CachedBasis{𝔽, <:AbstractBasis{𝔽}, <:FiberBundleBasisData},
    ) where {𝔽}
    xp1, xp2 = submanifold_components(M, p)
    zero_m = zero_vector(M.manifold, xp1)
    F = Fiber(M.manifold, xp1, M.type)
    zero_f = zero_vector(F, xp1)
    vs = typeof(ArrayPartition(zero_m, zero_f))[]
    for bv in get_vectors(M.manifold, xp1, B.data.base_basis)
        push!(vs, ArrayPartition(bv, zero_f))
    end
    for bv in get_vectors(F, xp2, B.data.fiber_basis)
        push!(vs, ArrayPartition(zero_m, bv))
    end
    return vs
end

"""
    setindex!(p::ArrayPartition, val, M::FiberBundle, s::Symbol)
    p[M::VectorBundle, s] = val

Set the element(s) at index `s` of a point `p` on a [`FiberBundle`](@ref) `M` to `val` by
using the symbols `:point` and `:fiber` or `:vector` for the base and fiber or vector
component, respectively.

!!! note

    The *content* of element of `p` is replaced, not the element itself.
"""
@inline function Base.setindex!(x::ArrayPartition, val, M::FiberBundle, s::Symbol)
    if s === :point
        return copyto!(x.x[1], val)
    elseif s === :vector || s === :fiber
        return copyto!(x.x[2], val)
    else
        throw(DomainError(s, "unknown component $s on $M."))
    end
end

function _vector_transport_direction(
        M::VectorBundle, p::ArrayPartition, X, d, m::FiberBundleProductVectorTransport,
    )
    px, pVx = submanifold_components(M.manifold, p)
    VXM, VXF = submanifold_components(M.manifold, X)
    dx, dVx = submanifold_components(M.manifold, d)
    return ArrayPartition(
        vector_transport_direction(M.manifold, px, VXM, dx, m.method_horizontal),
        bundle_transport_tangent_direction(M, px, pVx, VXF, dx, m.method_vertical),
    )
end

function _vector_transport_to(
        M::VectorBundle, p::ArrayPartition, X, q, m::FiberBundleProductVectorTransport,
    )
    px, pVx = submanifold_components(M.manifold, p)
    VXM, VXF = submanifold_components(M.manifold, X)
    qx, qVx = submanifold_components(M.manifold, q)
    return ArrayPartition(
        vector_transport_to(M.manifold, px, VXM, qx, m.method_horizontal),
        bundle_transport_tangent_to(M, px, pVx, VXF, qx, m.method_vertical),
    )
end

@inline function Base.view(x::ArrayPartition, M::FiberBundle, s::Symbol)
    (s === :point) && return x.x[1]
    (s === :vector || s === :fiber) && return x.x[2]
    throw(DomainError(s, "unknown component $s on $M."))
end
