# Shape spaces

Shape spaces are spaces of ``k`` points in ``ℝ^n`` up to simultaneous action of a group on all points.
The most commonly encountered are Kendall's pre-shape and shape spaces.
In the case of the Kendall's pre-shape spaces the action is translation and scaling.
In the case of the Kendall's shape spaces the action is translation, scaling and rotation.

```@example
using Manifolds, Plots

M = KendallsShapeSpace(2, 3)
# two random point on the shape space
p = [
    0.4385117672460505 -0.6877826444042382 0.24927087715818771
    -0.3830259932279294 0.35347460720654283 0.029551386021386548
]
q = [
    -0.42693314765896473 -0.3268567431952937 0.7537898908542584
    0.3054740561061169 -0.18962848284149897 -0.11584557326461796
]
# let's plot them as triples of points on a plane
fig = scatter(p[1,:], p[2,:], label="p", aspect_ratio=:equal)
scatter!(fig, q[1,:], q[2,:], label="q")

# aligning q to p
A = get_orbit_action(M)
a = optimal_alignment(A, p, q)
rot_q = apply(A, a, q)
scatter!(fig, rot_q[1,:], rot_q[2,:], label="q aligned to p")
```

A more extensive usage example is available in the `hand_gestures.jl` tutorial.

```@autodocs
Modules = [Manifolds, ManifoldsBase]
Pages = ["manifolds/KendallsPreShapeSpace.jl"]
Order = [:type]
```

```@autodocs
Modules = [Manifolds, ManifoldsBase]
Pages = ["manifolds/KendallsShapeSpace.jl"]
Order = [:type]
```

## Provided functions

```@autodocs
Modules = [Manifolds, ManifoldsBase]
Pages = ["manifolds/KendallsPreShapeSpace.jl"]
Order = [:function]
```

```@autodocs
Modules = [Manifolds, ManifoldsBase]
Pages = ["manifolds/KendallsShapeSpace.jl"]
Order = [:function]
```
