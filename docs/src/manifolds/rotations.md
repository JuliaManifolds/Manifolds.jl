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

```@raw html
<ul>
  <li id="Gallier2003">[<a>Gallier, 2003</a>]
    J. Gallier and D. Xu, “Computing exponentials of skew-symmetric matrices
    and logarithms of orthogonal matrices,” <i>International Journal of
    Robotics and Automation</i>, vol. 18, no. 1, pp. 10–20, 2003.
  </li>
  <li id="Andrica2013">[<a>Andrica, 2013</a>]
    D. Andrica and R.-A. Rohan, “Computing the Rodrigues coefficients of the
    exponential map of the Lie groups of matrices,” <i>Balkan Journal of
    Geometry and Its Applications</i>, vol. 18, no. 2, pp. 1–10, 2013.
  </li>
</ul>
```
