# Parametric surfaces

```@autodocs
Modules = [Manifolds, Base]
Pages = ["manifolds/ParametricSurface.jl"]
Order = [:type, :function]
Private=false
Public=true
```

## Example

The Möbius band is parametrized by a strip coordinate $u \in [-w, w]$ and an angle
$\theta \in \mathbb S^1$. Its parameter space is therefore the product of a
[`Hyperrectangle`](@ref) and a [`Circle`](@ref).

```@example mobius-band
using Manifolds, Plots, RecursiveArrayTools, ForwardDiff

half_width = 0.35
M_param = ProductManifold(Hyperrectangle([-half_width], [half_width]), Circle())
M_embed = Euclidean(3)

function mobius!(q, p)
    u, θ = p.x[1][1], p.x[2][]
    q .= [
        (1 + u * cos(θ / 2)) * cos(θ),
        (1 + u * cos(θ / 2)) * sin(θ),
        u * sin(θ / 2),
    ]
    return q
end

function inverse_mobius!(p, q)
    θ = atan(q[2], q[1])
    p.x[1][1] =
        (q[1] - cos(θ)) * cos(θ / 2) * cos(θ) +
        (q[2] - sin(θ)) * cos(θ / 2) * sin(θ) + q[3] * sin(θ / 2)
    p.x[2][] = θ
    return p
end

function jacobian_mobius!(J, p)
    ForwardDiff.jacobian!(J, x -> mobius!(Vector{eltype(x)}(undef, manifold_dimension(M_embed)), x), p)
    return J
end

M = ParametricSurface(M_param, M_embed, mobius!, inverse_mobius!, jacobian_mobius!)
```

Sampling its parametrization can be used to produce a three-dimensional view of the band.

```@example mobius-band
strip_coordinates = range(-half_width, half_width; length = 41)
angles = range(-π, π; length = 181)
xyz = [mobius!(zeros(3), ArrayPartition([u], [θ])) for u in strip_coordinates, θ in angles]
x = [p[1] for p in xyz]
y = [p[2] for p in xyz] 
z = [p[3] for p in xyz]
```

Which can then be visualized using for example `GLMakie.jl`:
`GLMakie.surface(x, y, z)`.

All the usual tools still work as in the [embedded torus example](../tutorials/working-in-charts.md).

## Internal docs

```@autodocs
Modules = [Manifolds, Base]
Pages = ["manifolds/ParametricSurface.jl"]
Order = [:type, :function]
Private=true
Public=false
```
