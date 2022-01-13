# Symplectic

The manifold [`Symplectic`](@ref) denoted, $\operatorname{Sp}(2n)$, represents transformations into symplectic subspaces which keep the
canonical symplectic form over $\mathbb{R}^{2n \times 2n }$ invariant under the standard embedding inner product.
The canonical symplectic form is $\omega(x, y) = x^T Q y$ for elements $x, y \in \mathbb{R}^{2n \times 2n }$, with
\\[
    Q = 
    \begin{bmatrix}
     0  &  I \\
    -I  &  0
    \end{bmatrix}.
\\] 
That means given an element $p \in \operatorname{Sp}{2n}$, 
\\[
    \omega (p x, p y) = x^T(p^TQp)y = x^TQy = \omega(x, y),
\\]
leading to the requirement on $p$ that $p^TQp = Q$.

```@autodocs
Modules = [Manifolds]
Pages = ["manifolds/Symplectic.jl"]
Order = [:type, :function]
```
