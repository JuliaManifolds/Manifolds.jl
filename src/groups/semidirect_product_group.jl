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

function show(io::IO, op::SemidirectProductOperation)
    print(io, "SemidirectProductOperation($(op.action), $(op.opN), $(op.opH))")
end

const SemidirectProductGroup{N,H,A,ON,OH} =
    GroupManifold{ProductManifold{Tuple{N,H}},SemidirectProductOperation{A,ON,OH}}

@doc doc"""
    SemidirectProductGroup(
        N::GroupManifold,
        H::GroupManifold,
        A::AbstractGroupAction,
    )

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
    base_group(A) == H || error("")
    g_manifold(A) == N || error("")
    op = SemidirectProductOperation(A, N.op, H.op)
    PG = ProductManifold(N, H)
    return GroupManifold(PG, op)
end

function show(io::IO, G::SemidirectProductGroup)
    PG = base_manifold(G)
    action = G.op.action
    print(
        io,
        "SemidirectProductGroup($(submanifold(PG, 1)), $(submanifold(PG, 2)), $(action))",
    )
end

function inv!(G::SemidirectProductGroup, y, x)
    PG = base_manifold(G)
    inv!(submanifold(PG, 2), submanifold_component(y, 2), submanifold_component(x, 2))
    ninv = inv(submanifold(PG, 1), submanifold_component(x, 1))
    apply!(G.op.action, submanifold_component(y, 1), submanifold_component(y, 2), ninv)
    return y
end
inv!(G::AG, y, e::Identity{AG}) where {AG<:SemidirectProductGroup} = identity!(G, y, e)

function identity!(G::SemidirectProductGroup, y, x)
    PG = base_manifold(G)
    for i in (1, 2)
        identity!(
            submanifold(PG, i),
            submanifold_component(y, i),
            submanifold_component(x, i),
        )
    end
    return y
end
identity!(G::GT, e::E, ::E) where {GT<:SemidirectProductGroup,E<:Identity{GT}} = e

function compose!(G::SemidirectProductGroup, z, x, y)
    PG = base_manifold(G)
    N = submanifold(PG, 1)
    H = submanifold(PG, 2)
    A = G.op.action
    compose!(H, submanifold_component.((z, x, y), 2)...)
    zₙtmp = apply(A, submanifold_component(x, 2), submanifold_component(y, 1))
    compose!(N, submanifold_component.((z, x), 1)..., zₙtmp)
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
function compose!(G::GT, z, e::E, ::E) where {GT<:SemidirectProductGroup,E<:Identity{GT}}
    identity!(G, z, e)
    return z
end

function translate_diff!(G::SemidirectProductGroup, vout, x, y, v, conv::LeftAction)
    PG = base_manifold(G)
    N = submanifold(PG, 1)
    H = submanifold(PG, 2)
    A = G.op.action
    nx, hx = submanifold_component.(Ref(x), (1, 2))
    ny, hy = submanifold_component.(Ref(y), (1, 2))
    nv, hv = submanifold_component.(Ref(v), (1, 2))
    nvout, hvout = submanifold_component.(Ref(vout), (1, 2))
    translate_diff!(H, hvout, hx, hy, hv, conv)
    nw = apply_diff(A, hx, ny, nv)
    nz = apply(A, hx, ny)
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
    #TODO: use zero_tangent_vector here
    vout = similar_result(base_manifold(G), translate_diff, x, y, v)
    translate_diff!(G, vout, x, y, v, conv)
    return vout
end
