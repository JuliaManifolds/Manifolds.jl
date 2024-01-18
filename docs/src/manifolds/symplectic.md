# Symplectic matrices

The [`SymplecticMatrices`](@ref) manifold, denoted $\operatorname{Sp}(2n, 𝔽)$, is a closed, embedded, submanifold of
$𝔽^{2n×2n}$ that represents transformations into symplectic subspaces which keep the
canonical symplectic form over $𝔽^{2n×2n}$ invariant under the standard embedding inner product.
The canonical symplectic form is a non-degenerate bilinear and skew symmetric map
$\omega\colon 𝔽 𝔽^{2n}×𝔽^{2n}
→ 𝔽$, given by
$\omega(x, y) = x^T Q_{2n} y$ for elements $x, y \in 𝔽^{2n}$, with
````math
    Q_{2n} =
    \begin{bmatrix}
     0_n  &  I_n \\
    -I_n  &  0_n
    \end{bmatrix}.
````
That means that an element $p \in \operatorname{Sp}(2n)$ must fulfill the requirement that
````math
    \omega (p x, p y) = x^T(p^TQp)y = x^TQy = \omega(x, y),
````
leading to the requirement on $p$ that $p^TQp = Q$.

The symplectic manifold also forms a group under matrix multiplication, called the $\textit{symplectic group}$.
Since all the symplectic matrices necessarily have determinant one, the [symplectic group](https://en.wikipedia.org/wiki/Symplectic_group)
$\operatorname{Sp}(2n, 𝔽)$ is a subgroup of the special linear group, $\operatorname{SL}(2n, 𝔽)$. When the underlying
field is either $ℝ$ or $ℂ$ the symplectic group with a manifold structure constitutes a Lie group, with the Lie
Algebra
````math
    \mathfrak{sp}(2n,F) = \{H \in 𝔽^{2n×2n} \;|\; Q H + H^{T} Q = 0\}.
````
This set is also known as the [Hamiltonian matrices](https://en.wikipedia.org/wiki/Hamiltonian_matrix), which have the
property that $(QH)^T = QH$ and are commonly used in physics.

```@autodocs
Modules = [Manifolds]
Pages = ["manifolds/Symplectic.jl"]
Order = [:type, :function]
```

## Literature

```@bibliography
Pages = ["symplectic.md"]
Canonical=false
```