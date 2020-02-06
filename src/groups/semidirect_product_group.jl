"""
    SemidirectProductOperation(action::AbstractGroupAction)

Group operation of a semidirect product group. The operation consists of the operation
`opN` on a normal subgroup `N`, the operation `opH` on a subgroup `H`, and an automorphism
`action` of elements of `H` on `N`. Only the action is stored.
"""
struct SemidirectProductOperation{A<:AbstractGroupAction} <: AbstractGroupOperation
    action::A
end

function show(io::IO, op::SemidirectProductOperation)
    print(io, "SemidirectProductOperation($(op.action))")
end

const SemidirectProductGroup{N,H,A} =
    GroupManifold{ProductManifold{Tuple{N,H}},SemidirectProductOperation{A}}

@doc raw"""
    SemidirectProductGroup(N::GroupManifold, H::GroupManifold, A::AbstractGroupAction)

A group that is the semidirect product of a normal group $\mathcal{N}$ and a subgroup
$\mathcal{H}$, written $\mathcal{G} = \mathcal{N} ⋊_θ \mathcal{H}$, where
$θ: \mathcal{H} × \mathcal{N} → \mathcal{N}$ is an automorphism action of $\mathcal{H}$ on
$\mathcal{N}$. The group $\mathcal{G}$ has the composition rule

````math
g \circ g' = (n, h) \circ (n', h') = (n \circ θ_h(n'), h \circ h')
````

and the inverse

````math
g^{-1} = (n, h)^{-1} = (θ_{h^{-1}}(n^{-1}), h^{-1}).
````
"""
function SemidirectProductGroup(N::GroupManifold, H::GroupManifold, A::AbstractGroupAction)
    N === g_manifold(A) || error("Subgroup $(N) must be the G-manifold of action $(A)")
    H === base_group(A) || error("Subgroup $(H) must be the base group of action $(A)")
    op = SemidirectProductOperation(A)
    M = ProductManifold(N, H)
    return GroupManifold(M, op)
end

function show(io::IO, G::SemidirectProductGroup)
    M = base_manifold(G)
    N, H = M.manifolds
    A = G.op.action
    print(io, "SemidirectProductGroup($(N), $(H), $(A))")
end

submanifold(G::SemidirectProductGroup, i) = submanifold(base_manifold(G), i)

_padpoint!(G::SemidirectProductGroup, q) = q

_padvector!(G::SemidirectProductGroup, X) = X

inv(G::GT, e::Identity{GT}) where {GT<:SemidirectProductGroup} = e

function inv!(G::SemidirectProductGroup, q, p)
    M = base_manifold(G)
    N, H = M.manifolds
    A = G.op.action
    nx, hx = submanifold_components(G, p)
    ny, hy = submanifold_components(G, q)
    inv!(H, hy, hx)
    ninv = inv(N, nx)
    apply!(A, ny, hy, ninv)
    @inbounds _padpoint!(G, q)
    return q
end
inv!(G::AG, p, e::Identity{AG}) where {AG<:SemidirectProductGroup} = identity!(G, p, e)

identity(G::GT, e::Identity{GT}) where {GT<:SemidirectProductGroup} = e

function identity!(G::SemidirectProductGroup, q, p)
    M = base_manifold(G)
    N, H = M.manifolds
    nx, hx = submanifold_components(G, p)
    ny, hy = submanifold_components(G, q)
    identity!(N, ny, nx)
    identity!(H, hy, hx)
    @inbounds _padpoint!(G, q)
    return q
end
identity!(G::GT, e::E, ::E) where {GT<:SemidirectProductGroup,E<:Identity{GT}} = e

compose(G::GT, p, e::Identity{GT}) where {GT<:SemidirectProductGroup} = p
compose(G::GT, e::Identity{GT}, p) where {GT<:SemidirectProductGroup} = p
compose(G::GT, e::E, ::E) where {GT<:SemidirectProductGroup,E<:Identity{GT}} = e

function compose!(G::SemidirectProductGroup, x, p, q)
    M = base_manifold(G)
    N, H = M.manifolds
    A = G.op.action
    nx, hx = submanifold_components(G, p)
    ny, hy = submanifold_components(G, q)
    nz, hz = submanifold_components(G, x)
    compose!(H, hz, hx, hy)
    zₙtmp = apply(A, hx, ny)
    compose!(N, nz, nx, zₙtmp)
    @inbounds _padpoint!(G, x)
    return x
end
compose!(G::GT, x, ::Identity{GT}, q) where {GT<:SemidirectProductGroup} = copyto!(x, q)
compose!(G::GT, x, p, ::Identity{GT}) where {GT<:SemidirectProductGroup} = copyto!(x, p)
function compose!(G::GT, x, e::E, ::E) where {GT<:SemidirectProductGroup,E<:Identity{GT}}
    return identity!(G, x, e)
end

function translate_diff!(G::SemidirectProductGroup, Y, p, q, X, conv::LeftAction)
    M = base_manifold(G)
    N, H = M.manifolds
    A = G.op.action
    np, hp = submanifold_components(G, p)
    nq, hq = submanifold_components(G, q)
    nX, hX = submanifold_components(G, X)
    nY, hY = submanifold_components(G, Y)
    translate_diff!(H, hY, hp, hq, hX, conv)
    nZ = apply_diff(A, hp, nq, nX)
    nr = apply(A, hp, nq)
    translate_diff!(N, nY, np, nr, nZ, conv)
    @inbounds _padvector!(G, Y)
    return Y
end

function hat!(G::SemidirectProductGroup, Y, p, X)
    M = base_manifold(G)
    N, H = M.manifolds
    dimN = manifold_dimension(N)
    dimH = manifold_dimension(H)
    @assert length(X) == dimN + dimH
    nx, hx = submanifold_components(G, p)
    nV, hV = submanifold_components(G, Y)
    hat!(N, nV, nx, view(X, 1:dimN))
    hat!(H, hV, hx, view(X, dimN+1:dimN+dimH))
    @inbounds _padvector!(G, Y)
    return Y
end

function vee!(G::SemidirectProductGroup, Y, p, X)
    M = base_manifold(G)
    N, H = M.manifolds
    dimN = manifold_dimension(N)
    dimH = manifold_dimension(H)
    @assert length(Y) == dimN + dimH
    nx, hx = submanifold_components(G, p)
    nV, hV = submanifold_components(G, X)
    vee!(N, view(Y, 1:dimN), nx, nV)
    vee!(H, view(Y, dimN+1:dimN+dimH), hx, hV)
    return Y
end

function zero_tangent_vector(G::SemidirectProductGroup, p)
    X = allocate_result(G, zero_tangent_vector, p)
    zero_tangent_vector!(G, X, p)
    return X
end

function zero_tangent_vector!(G::SemidirectProductGroup, X, p)
    M = base_manifold(G)
    N, H = M.manifolds
    nx, hx = submanifold_components(G, p)
    nv, hv = submanifold_components(G, X)
    zero_tangent_vector!(N, nv, nx)
    zero_tangent_vector!(H, hv, hx)
    return X
end

function isapprox(G::SemidirectProductGroup, p, q; kwargs...)
    M = base_manifold(G)
    N, H = M.manifolds
    nx, hx = submanifold_components(G, p)
    ny, hy = submanifold_components(G, q)
    return isapprox(N, nx, ny; kwargs...) && isapprox(H, hx, hy; kwargs...)
end
function isapprox(G::SemidirectProductGroup, p, X, Y; kwargs...)
    M = base_manifold(G)
    N, H = M.manifolds
    nx, hx = submanifold_components(G, p)
    nv, hv = submanifold_components(G, X)
    nw, hw = submanifold_components(G, Y)
    return isapprox(N, nx, nv, nw; kwargs...) && isapprox(H, hx, hv, hw; kwargs...)
end
function isapprox(G::GT, p, e::Identity{GT}; kwargs...) where {GT<:SemidirectProductGroup}
    return isapprox(G, e, p; kwargs...)
end
function isapprox(G::GT, e::Identity{GT}, p; kwargs...) where {GT<:SemidirectProductGroup}
    return isapprox(G, identity(G, p), p; kwargs...)
end
function isapprox(
    ::GT,
    ::E,
    ::E;
    kwargs...,
) where {GT<:SemidirectProductGroup,E<:Identity{GT}}
    return true
end
