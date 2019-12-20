# Rotations

The manifold $\mathrm{SO}(n)$ of orthogonal matrices with determinant $+1$ in $\mathbb R^{n\times n}$, i.e.

$\mathrm{SO}(n) = \bigl\{R \in \mathbb{R}^{n\times n} \big| RR^{\mathrm{T}} =
R^{\mathrm{T}}R = \mathrm{I}_n, \det(R) = 1 \bigr\}$

 $\mathrm{SO}(n)$ is a subgroup of the orthogonal group $\mathrm{O}(n)$ and also known as the special orthogonal group or the set of rotations group.

Tangent vectors are represented by elements of the corresponding Lie algebra, which is the tangent space at the identity element. This convention allows for more efficient operations on tangent vectors. Tangent spaces at different points are different vector spaces.

Let $L_R\colon \mathrm{SO}(n) \to \mathrm{SO}(n)$ where $R \in \mathrm{SO}(n)$ be the left-multiplication by $R$, that is $L_R(S) = RS$. The tangent space at rotation $R$, $T_R \mathrm{SO}(n)$, is related to the tangent space at the identity rotation $\mathrm{I}_n$ by the differential of $L_R$ at identity, $(\mathrm{d}L_R)_{\mathrm{I}_n} \colon T_{\mathrm{I}_n} \mathrm{SO}(n) \to T_R \mathrm{SO}(n)$. For a tangent vector at the identity rotation $v \in T_{\mathrm{I}_n} \mathrm{SO}(n)$ the matrix representation of the corresponding tangent vector $w$ at a rotation $R$ can be obtained by matrix multiplication: $w=Rv \in T_R \mathrm{SO}(n)$. You can compare the functions [`log!(::Manifolds.Rotations, v, x, y)`](@ref) and [`exp!(::Manifolds.Rotations, y, x, v)`](@ref) to see how it works in practice.

```@autodocs
Modules = [Manifolds]
Pages = ["Rotations.jl"]
Order = [:type, :function]
```

## Literature
