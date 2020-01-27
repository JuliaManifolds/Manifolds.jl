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

@doc doc"""
    SemidirectProductGroup(N::GroupManifold, H::GroupManifold, A::AbstractGroupAction)

A group that is the semidirect product of a normal group $N$ and a subgroup $H$, written
$G = N ⋊_θ H$, where $θ: H × N \to N$ is an automorphism action
of $H$ on $N$. The group $G$ has the composition rule

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

_padpoint!(G::SemidirectProductGroup, y) = y

_padvector!(G::SemidirectProductGroup, v) = v

inv(G::GT, e::Identity{GT}) where {GT<:SemidirectProductGroup} = e

function inv!(G::SemidirectProductGroup, y, x)
    M = base_manifold(G)
    N, H = M.manifolds
    A = G.op.action
    nx, hx = submanifold_components(G, x)
    ny, hy = submanifold_components(G, y)
    inv!(H, hy, hx)
    ninv = inv(N, nx)
    apply!(A, ny, hy, ninv)
    @inbounds _padpoint!(G, y)
    return y
end
inv!(G::AG, y, e::Identity{AG}) where {AG<:SemidirectProductGroup} = identity!(G, y, e)

identity(G::GT, e::Identity{GT}) where {GT<:SemidirectProductGroup} = e

function identity!(G::SemidirectProductGroup, y, x)
    M = base_manifold(G)
    N, H = M.manifolds
    nx, hx = submanifold_components(G, x)
    ny, hy = submanifold_components(G, y)
    identity!(N, ny, nx)
    identity!(H, hy, hx)
    @inbounds _padpoint!(G, y)
    return y
end
identity!(G::GT, e::E, ::E) where {GT<:SemidirectProductGroup,E<:Identity{GT}} = e

compose(G::GT, x, e::Identity{GT}) where {GT<:SemidirectProductGroup} = x
compose(G::GT, e::Identity{GT}, x) where {GT<:SemidirectProductGroup} = x
compose(G::GT, e::E, ::E) where {GT<:SemidirectProductGroup,E<:Identity{GT}} = e

function compose!(G::SemidirectProductGroup, z, x, y)
    M = base_manifold(G)
    N, H = M.manifolds
    A = G.op.action
    nx, hx = submanifold_components(G, x)
    ny, hy = submanifold_components(G, y)
    nz, hz = submanifold_components(G, z)
    compose!(H, hz, hx, hy)
    zₙtmp = apply(A, hx, ny)
    compose!(N, nz, nx, zₙtmp)
    @inbounds _padpoint!(G, z)
    return z
end
compose!(G::GT, z, ::Identity{GT}, y) where {GT<:SemidirectProductGroup} = copyto!(z, y)
compose!(G::GT, z, x, ::Identity{GT}) where {GT<:SemidirectProductGroup} = copyto!(z, x)
function compose!(G::GT, z, e::E, ::E) where {GT<:SemidirectProductGroup,E<:Identity{GT}}
    return identity!(G, z, e)
end

function translate_diff!(G::SemidirectProductGroup, vout, x, y, v, conv::LeftAction)
    M = base_manifold(G)
    N, H = M.manifolds
    A = G.op.action
    nx, hx = submanifold_components(G, x)
    ny, hy = submanifold_components(G, y)
    nv, hv = submanifold_components(G, v)
    nvout, hvout = submanifold_components(G, vout)
    translate_diff!(H, hvout, hx, hy, hv, conv)
    nw = apply_diff(A, hx, ny, nv)
    nz = apply(A, hx, ny)
    translate_diff!(N, nvout, nx, nz, nw, conv)
    @inbounds _padvector!(G, vout)
    return vout
end

function hat!(G::SemidirectProductGroup, V, x, v)
    M = base_manifold(G)
    N, H = M.manifolds
    dimN = manifold_dimension(N)
    dimH = manifold_dimension(H)
    @assert length(v) == dimN + dimH
    nx, hx = submanifold_components(G, x)
    nV, hV = submanifold_components(G, V)
    hat!(N, nV, nx, view(v, 1:dimN))
    hat!(H, hV, hx, view(v, dimN+1:dimN+dimH))
    @inbounds _padvector!(G, V)
    return V
end

function vee!(G::SemidirectProductGroup, v, x, V)
    M = base_manifold(G)
    N, H = M.manifolds
    dimN = manifold_dimension(N)
    dimH = manifold_dimension(H)
    @assert length(v) == dimN + dimH
    nx, hx = submanifold_components(G, x)
    nV, hV = submanifold_components(G, V)
    vee!(N, view(v, 1:dimN), nx, nV)
    vee!(H, view(v, dimN+1:dimN+dimH), hx, hV)
    return v
end

function zero_tangent_vector(G::SemidirectProductGroup, x)
    v = allocate_result(G, zero_tangent_vector, x)
    zero_tangent_vector!(G, v, x)
    return v
end

function zero_tangent_vector!(G::SemidirectProductGroup, v, x)
    M = base_manifold(G)
    N, H = M.manifolds
    nx, hx = submanifold_components(G, x)
    nv, hv = submanifold_components(G, v)
    zero_tangent_vector!(N, nv, nx)
    zero_tangent_vector!(H, hv, hx)
    return v
end

function isapprox(G::SemidirectProductGroup, x, y; kwargs...)
    M = base_manifold(G)
    N, H = M.manifolds
    nx, hx = submanifold_components(G, x)
    ny, hy = submanifold_components(G, y)
    return isapprox(N, nx, ny; kwargs...) && isapprox(H, hx, hy; kwargs...)
end
function isapprox(G::SemidirectProductGroup, x, v, w; kwargs...)
    M = base_manifold(G)
    N, H = M.manifolds
    nx, hx = submanifold_components(G, x)
    nv, hv = submanifold_components(G, v)
    nw, hw = submanifold_components(G, w)
    return isapprox(N, nx, nv, nw; kwargs...) && isapprox(H, hx, hv, hw; kwargs...)
end
function isapprox(G::GT, x, e::Identity{GT}; kwargs...) where {GT<:SemidirectProductGroup}
    return isapprox(G, e, x; kwargs...)
end
function isapprox(G::GT, e::Identity{GT}, x; kwargs...) where {GT<:SemidirectProductGroup}
    return isapprox(G, identity(G, x), x; kwargs...)
end
function isapprox(
    ::GT,
    ::E,
    ::E;
    kwargs...,
) where {GT<:SemidirectProductGroup,E<:Identity{GT}}
    return true
end
