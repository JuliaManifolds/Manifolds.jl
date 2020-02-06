# Rotations

The manifold $\mathrm{SO}(n)$ of orthogonal matrices with determinant $+1$ in $ℝ^{n × n}$, i.e.

$\mathrm{SO}(n) = \bigl\{R ∈ ℝ^{n × n} \big| R R^{\mathrm{T}} =
R^{\mathrm{T}}R = I_n, \det(R) = 1 \bigr\}$

The Lie group $\mathrm{SO}(n)$ is a subgroup of the orthogonal group $\mathrm{O}(n)$ and also known as the special orthogonal group or the set of rotations group.
See also [`SpecialOrthogonal`](@ref), which is this manifold equipped with the group operation.

Tangent vectors are represented by elements of the corresponding Lie algebra, which is the tangent space at the identity element.
This convention allows for more efficient operations on tangent vectors.
Tangent spaces at different points are different vector spaces.

Let $L_R: \mathrm{SO}(n) → \mathrm{SO}(n)$ where $R ∈ \mathrm{SO}(n)$ be the left-multiplication by $R$, that is $L_R(S) = RS$.
The tangent space at rotation $R$, $T_R \mathrm{SO}(n)$, is related to the tangent space at the identity rotation $I_n$ by the differential of $L_R$ at identity, $(\mathrm{d}L_R)_{I_n} : T_{I_n} \mathrm{SO}(n) → T_R \mathrm{SO}(n)$.
For a tangent vector at the identity rotation $X ∈ T_{I_n} \mathrm{SO}(n)$ the matrix representation of the corresponding tangent vector $Y$ at a rotation $R$ can be obtained by matrix multiplication: $Y = RX ∈ T_R \mathrm{SO}(n)$.
You can compare the functions [`log!(::Manifolds.Rotations, X, p, q)`](@ref) and [`exp!(::Manifolds.Rotations, q, p, X)`](@ref) to see how it works in practice.

```@autodocs
Modules = [Manifolds]
Pages = ["manifolds/Rotations.jl"]
Order = [:type, :function]
```

## Literature
