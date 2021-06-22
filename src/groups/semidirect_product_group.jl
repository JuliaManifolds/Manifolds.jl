"""
    SemidirectProductOperation(action::AbstractGroupAction)

Group operation of a semidirect product group. The operation consists of the operation
`opN` on a normal subgroup `N`, the operation `opH` on a subgroup `H`, and an automorphism
`action` of elements of `H` on `N`. Only the action is stored.
"""
struct SemidirectProductOperation{A<:AbstractGroupAction} <: AbstractGroupOperation
    action::A
end

function Base.show(io::IO, op::SemidirectProductOperation)
    return print(io, "SemidirectProductOperation($(op.action))")
end

const SemidirectProductGroup{𝔽,N,H,A} =
    GroupManifold{𝔽,ProductManifold{𝔽,Tuple{N,H}},SemidirectProductOperation{A}}

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
function SemidirectProductGroup(
    N::GroupManifold{𝔽},
    H::GroupManifold{𝔽},
    A::AbstractGroupAction,
) where {𝔽}
    N === g_manifold(A) || error("Subgroup $(N) must be the G-manifold of action $(A)")
    H === base_group(A) || error("Subgroup $(H) must be the base group of action $(A)")
    op = SemidirectProductOperation(A)
    M = ProductManifold(N, H)
    return GroupManifold(M, op)
end

function Base.show(io::IO, G::SemidirectProductGroup)
    M = base_manifold(G)
    N, H = M.manifolds
    A = G.op.action
    return print(io, "SemidirectProductGroup($(N), $(H), $(A))")
end

submanifold(G::SemidirectProductGroup, i) = submanifold(base_manifold(G), i)

_padpoint!(G::SemidirectProductGroup, q) = q

_padvector!(G::SemidirectProductGroup, X) = X

Base.inv(G::GT, e::Identity{GT}) where {GT<:SemidirectProductGroup} = e

function inv!(G::SemidirectProductGroup, q, p)
    M = base_manifold(G)
    N, H = M.manifolds
    A = G.op.action
    np, hp = submanifold_components(G, p)
    nq, hq = submanifold_components(G, q)
    inv!(H, hq, hp)
    npinv = inv(N, np)
    apply!(A, nq, hq, npinv)
    @inbounds _padpoint!(G, q)
    return q
end
inv!(G::AG, p, e::Identity{AG}) where {AG<:SemidirectProductGroup} = identity!(G, p, e)

Base.identity(G::GT, e::Identity{GT}) where {GT<:SemidirectProductGroup} = e

function identity!(G::SemidirectProductGroup, q, p)
    M = base_manifold(G)
    N, H = M.manifolds
    np, hp = submanifold_components(G, p)
    nq, hq = submanifold_components(G, q)
    identity!(N, nq, np)
    identity!(H, hq, hp)
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
    np, hp = submanifold_components(G, p)
    nq, hq = submanifold_components(G, q)
    nx, hx = submanifold_components(G, x)
    compose!(H, hx, hp, hq)
    nxtmp = apply(A, hp, nq)
    compose!(N, nx, np, nxtmp)
    @inbounds _padpoint!(G, x)
    return x
end
compose!(G::GT, x, ::Identity{GT}, q) where {GT<:SemidirectProductGroup} = copyto!(x, q)
compose!(G::GT, x, p, ::Identity{GT}) where {GT<:SemidirectProductGroup} = copyto!(x, p)
function compose!(G::GT, x, e::E, ::E) where {GT<:SemidirectProductGroup,E<:Identity{GT}}
    return identity!(G, x, e)
end

@doc raw"""
    translate_diff!(G::SemidirectProductGroup, Y, p, q, X, conX::LeftAction)

Perform differential of the left translation on the semidirect product group `G`.

Since the left translation is defined as (cf. [`SemidirectProductGroup`](@ref)):

````math
L_{(n', h')} (n, h) = ( L_{n'} θ_{h'}(n), L_{h'} h)
````

then its differential can be computed as

````math
\mathrm{d}L_{(n', h')}(X_n, X_h) = ( \mathrm{d}L_{n'} (\mathrm{d}θ_{h'}(X_n)), \mathrm{d}L_{h'} X_h).
````
"""
function translate_diff!(G::SemidirectProductGroup, Y, p, q, X, conX::LeftAction)
    M = base_manifold(G)
    N, H = M.manifolds
    A = G.op.action
    np, hp = submanifold_components(G, p)
    nq, hq = submanifold_components(G, q)
    nX, hX = submanifold_components(G, X)
    nY, hY = submanifold_components(G, Y)
    translate_diff!(H, hY, hp, hq, hX, conX)
    nZ = apply_diff(A, hp, nq, nX)
    nr = apply(A, hp, nq)
    translate_diff!(N, nY, np, nr, nZ, conX)
    @inbounds _padvector!(G, Y)
    return Y
end

function get_vector!(G::SemidirectProductGroup, Y, p, X, B::VeeOrthogonalBasis)
    M = base_manifold(G)
    N, H = M.manifolds
    dimN = manifold_dimension(N)
    dimH = manifold_dimension(H)
    @assert length(X) == dimN + dimH
    np, hp = submanifold_components(G, p)
    nY, hY = submanifold_components(G, Y)
    get_vector!(N, nY, np, view(X, 1:dimN), B)
    get_vector!(H, hY, hp, view(X, (dimN + 1):(dimN + dimH)), B)
    @inbounds _padvector!(G, Y)
    return Y
end
eval(
    quote
        @invoke_maker 1 AbstractManifold get_vector!(
            M::SemidirectProductGroup,
            Xⁱ,
            e::Identity,
            X,
            B::VeeOrthogonalBasis,
        )
    end,
)

function get_coordinates!(G::SemidirectProductGroup, Y, p, X, B::VeeOrthogonalBasis)
    M = base_manifold(G)
    N, H = M.manifolds
    dimN = manifold_dimension(N)
    dimH = manifold_dimension(H)
    @assert length(Y) == dimN + dimH
    np, hp = submanifold_components(G, p)
    nY, hY = submanifold_components(G, X)
    get_coordinates!(N, view(Y, 1:dimN), np, nY, B)
    get_coordinates!(H, view(Y, (dimN + 1):(dimN + dimH)), hp, hY, B)
    return Y
end
eval(
    quote
        @invoke_maker 1 AbstractManifold get_coordinates!(
            M::SemidirectProductGroup,
            Y,
            e::Identity,
            X,
            B::VeeOrthogonalBasis,
        )
    end,
)

function zero_vector(G::SemidirectProductGroup, p)
    X = allocate_result(G, zero_vector, p)
    zero_vector!(G, X, p)
    return X
end

function zero_vector!(G::SemidirectProductGroup, X, p)
    M = base_manifold(G)
    N, H = M.manifolds
    np, hp = submanifold_components(G, p)
    nX, hX = submanifold_components(G, X)
    zero_vector!(N, nX, np)
    zero_vector!(H, hX, hp)
    return X
end

function Base.isapprox(G::SemidirectProductGroup, p, q; kwargs...)
    M = base_manifold(G)
    N, H = M.manifolds
    np, hp = submanifold_components(G, p)
    nq, hq = submanifold_components(G, q)
    return isapprox(N, np, nq; kwargs...) && isapprox(H, hp, hq; kwargs...)
end
function Base.isapprox(G::SemidirectProductGroup, p, X, Y; kwargs...)
    M = base_manifold(G)
    N, H = M.manifolds
    np, hp = submanifold_components(G, p)
    nX, hX = submanifold_components(G, X)
    nY, hY = submanifold_components(G, Y)
    return isapprox(N, np, nX, nY; kwargs...) && isapprox(H, hp, hX, hY; kwargs...)
end
function Base.isapprox(
    G::GT,
    p,
    e::Identity{GT};
    kwargs...,
) where {GT<:SemidirectProductGroup}
    return isapprox(G, e, p; kwargs...)
end
function Base.isapprox(
    G::GT,
    e::Identity{GT},
    p;
    kwargs...,
) where {GT<:SemidirectProductGroup}
    return isapprox(G, identity(G, p), p; kwargs...)
end
function Base.isapprox(
    ::GT,
    ::E,
    ::E;
    kwargs...,
) where {GT<:SemidirectProductGroup,E<:Identity{GT}}
    return true
end
