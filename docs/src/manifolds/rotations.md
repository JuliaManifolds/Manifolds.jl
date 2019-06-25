# Rotations

The manifold $\mathrm{SO}(n)$ of orthogonal matrices with determinant $+1$ in $\mathbb R^{n\times n}$, i.e.

$\mathrm{SO}(n) = \bigl\{R \in \mathbb{R}^{n\times n} \big| RR^{\mathrm{T}} =
R^{\mathrm{T}}R = \mathrm{I}_n, \operatorname{det}(R) = 1 \bigr\}$

The $\mathrm{SO}(n)$ is a subgroup of the orthogonal group $\mathrm{O}(n)$ and also known as the special orthogonal group or the set of rotations group.

Tangent vectors are represented by elements of the corresponding Lie algebra, that is the tangent space at the identity element. This convention allows for more efficient operations on tangent vectors, although note that in such tangent vectors are not orthogonal to rotations the are tangent at in the ambient space of matrices. Tangent spaces at different points are different vector spaces.

```@autodocs
Modules = [ManifoldMuseum]
Pages = ["Rotations.jl"]
Order = [:type, :function]
```
