"""
    SemidirectProductOperation(
        action::AbstractGroupAction,
        opN::AbstractGroupOperation,
        oph::AbstractGroupOperation,
    )

Group operation of a semidirect product group. The operation consists of the operation
`opN` on a normal subgroup `N`, the operation `opH` on a subgroup `H`, and an automorphism
`action` of elements of `H` on `N`.
"""
struct SemidirectProductOperation{
    A<:AbstractGroupAction,
    ON<:AbstractGroupOperation,
    OH<:AbstractGroupOperation,
} <: AbstractGroupOperation
    action::A
    opN::ON
    opH::OH
end

const SemidirectProductGroup{N,H,A,ON,OH} = GroupManifold{
    ProductManifold{Tuple{N,H}},
    SemidirectProductOperation{A,ON,OH},
}

@doc doc"""
    SemidirectProductGroup(
        N::GroupManifold,
        H::GroupManifold,
        A::AbstractGroupAction,
    )

A group that is the semidirect product of a normal group $N$ and a subgroup $H$, written
$G = N ⋊_{\theta} H$, where $θ: H × N \to N$ is an automorphism action
of $H$ on $N$. The group $G$ has the composition rule

````math
g \circ g′ = (n, h) \circ (n′, h′) = (n \circ θ_h(n′), h \circ h′)
````

and the inverse

````math
g^{-1} = (n, h)^{-1} = (θ_{h^{-1}}(n^{-1}), h^{-1}).
````
"""
function SemidirectProductGroup(
    N::GroupManifold,
    H::GroupManifold,
    A::AbstractGroupAction,
)
    base_group(A) == H || error("")
    g_manifold(A) == N || error("")
    op = SemidirectProductOperation(A, N.op, H.op)
    PG = ProductManifold(N, H)
    return GroupManifold(PG, op)
end

function submanifold_component(e::Identity{<:SemidirectProductOperation}, i::Integer)
    return Identity(submanifold(base_manifold(e.group), Val(i)))
end

function inv!(G::SemidirectProductGroup, y, x)
    PG = base_manifold(G)
    N = submanifold(PG, 1)
    H = submanifold(PG, 2)
    A = G.op.action
    inv!(H, y.parts[2], x.parts[2])
    ninv = inv(N, x.parts[1])
    apply!(A, y.parts[1], y.parts[2], ninv)
    return y
end
function inv!(G::AG, y, e::Identity{AG}) where {AG<:SemidirectProductGroup}
    PG = base_manifold(G)
    es = map(Identity, PG.manifolds)
    map(inv!, PG.manifolds, y.parts, es)
    return y
end

function identity!(G::SemidirectProductGroup, y, x)
    PG = base_manifold(G)
    map(identity!, PG.manifolds, y.parts, x.parts)
    return y
end
function identity!(G::GT, y, x::Identity{GT}) where {GT<:SemidirectProductGroup}
    PG = base_manifold(G)
    N = submanifold(PG, 1)
    H = submanifold(PG, 2)
    identity!(N, y.parts[1], Identity(N))
    identity!(H, y.parts[2], Identity(H))
    return y
end
identity!(G::GT, e::Identity{GT}, ::Identity{GT}) where {GT<:SemidirectProductGroup} = e

function compose!(G::SemidirectProductGroup, z, x, y)
    PG = base_manifold(G)
    N = submanifold(PG, 1)
    H = submanifold(PG, 2)
    A = G.op.action
    compose!(H, z.parts[2], x.parts[2], y.parts[2])
    zₙtmp = apply(A, x.parts[2], y.parts[1])
    compose!(N, z.parts[1], x.parts[1], zₙtmp)
    return z
end
function compose!(
    G::GT,
    z,
    e::Identity{GT},
    ::Identity{GT},
) where {GT<:SemidirectProductGroup}
    identity!(G, z, e)
    return z
end
function compose!(G::GT, z, ::Identity{GT}, y) where {GT<:SemidirectProductGroup}
    copyto!(z, y)
    return z
end
function compose!(G::GT, z, x, ::Identity{GT}) where {GT<:SemidirectProductGroup}
    copyto!(z, x)
    return z
end

function translate_diff!(G::SemidirectProductGroup, vout, x, y, v, conv::LeftAction)
    PG = base_manifold(G)
    N = submanifold(PG, 1)
    H = submanifold(PG, 2)
    A = G.op.action
    nx, hx, ny, hy = x.parts[1], x.parts[2], y.parts[1], y.parts[2]
    nvout, hvout, nv, hv = vout.parts[1], vout.parts[2], v.parts[1], v.parts[2]
    translate_diff!(H, hvout, hx, hy, hv, conv)
    nw = action_diff(A, hx, ny, nv, conv)
    nz = action(A, hx, ny, conv)
    translate_diff!(N, nvout, nx, nz, nw, conv)
    return vout
end

function translate_diff(
    G::SemidirectProductGroup,
    x,
    y,
    v,
    conv::ActionDirection = LeftAction(),
)
    vout = similar_result(base_manifold(G), x, y, v)
    translate_diff!(G, vout, x, y, v, conv)
    return vout
end
