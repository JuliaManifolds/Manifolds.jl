# Rotations

The manifold $\mathrm{SO}(n)$ of orthogonal matrices with determinant $+1$ in $ℝ^{n × n}$, i.e.

$\mathrm{SO}(n) = \bigl\{R ∈ ℝ^{n × n} \big| R R^{\mathrm{T}} =
R^{\mathrm{T}}R = I_n, \det(R) = 1 \bigr\}$

The Lie group $\mathrm{SO}(n)$ is a subgroup of the orthogonal group $\mathrm{O}(n)$ and also known as the special orthogonal group or the set of rotations group.
See also [`SpecialOrthogonal`](@ref), which is this manifold equipped with the group operation.

The tangent space to a point ``p ∈ \mathrm{SO}(n)`` is given by

```math
T_p\mathrm{SO}(n) = \{X : X=pY,\qquad Y=-Y^{\mathrm{T}}\},
```

i.e. all vectors that are a product of a skew symmetric matrix multiplied with ``p``.

Since the orthogonal matrices ``\mathrm{SO}(n)`` are a Lie group, tangent vectors can also be
represented by elements of the corresponding Lie algebra, which is the tangent space at the identity element.
In the notation above, this means we just store the component ``Y`` of ``X``.

This convention allows for more efficient operations on tangent vectors.
Tangent spaces at different points are different vector spaces.

Let $L_R: \mathrm{SO}(n) → \mathrm{SO}(n)$ where $R ∈ \mathrm{SO}(n)$ be the left-multiplication by $R$, that is $L_R(S) = RS$.
The tangent space at rotation $R$, $T_R \mathrm{SO}(n)$, is related to the tangent space at the identity rotation $I_n$ by the differential of $L_R$ at identity, $(\mathrm{d}L_R)_{I_n} : T_{I_n} \mathrm{SO}(n) → T_R \mathrm{SO}(n)$.
To convert the tangent vector representation at the identity rotation $X ∈ T_{I_n} \mathrm{SO}(n)$ (i.e., the default) to the matrix representation of the corresponding tangent vector $Y$ at a rotation $R$ use the [`embed`](@ref embed(::Manifolds.Rotations, :Any...)) which implements the following multiplication: $Y = RX ∈ T_R \mathrm{SO}(n)$.
You can compare the functions [`log`](@ref log(::Manifolds.Rotations, :Any...)) and [`exp`](@ref exp(::Manifolds.Rotations, ::Any...)) to see how it works in practice.

```@autodocs
Modules = [Manifolds]
Pages = ["manifolds/Rotations.jl"]
Order = [:type, :function]
```

## Literature
