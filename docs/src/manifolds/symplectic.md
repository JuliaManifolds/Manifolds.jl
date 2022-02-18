# Symplectic 

The [`Symplectic`](@ref) manifold, denoted $\operatorname{Sp}(2n, \mathbb{F})$, is a closed, embedded, submanifold of 
$\mathbb{F}^{2n \times 2n}$ that represents transformations into symplectic subspaces which keep the
canonical symplectic form over $\mathbb{F}^{2n \times 2n }$ invariant under the standard embedding inner product.
The canonical symplectic form is a non-degenerate bilinear and skew symmetric map 
$\omega\colon \mathbb{F}^{2n} \times \mathbb{F}^{2n} 
\rightarrow \mathbb{F}$, given by
$\omega(x, y) = x^T Q_{2n} y$ for elements $x, y \in \mathbb{F}^{2n}$, with
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
$\operatorname{Sp}(2n, \mathbb{F})$ is a subgroup of the special linear group, $\operatorname{SL}(2n, \mathbb{F})$. When the underlying 
field is either $\mathbb{R}$ or $\mathbb{C}$ the symplectic group with a manifold structure constitutes a Lie group, with the Lie
Algebra 
````math
    \mathfrak{sp}(2n,F) = \{H \in \mathbb{F}^{2n \times 2n} \;|\; Q H + H^{T} Q = 0\}.
````
This set is also known as the [Hamiltonian matrices](https://en.wikipedia.org/wiki/Hamiltonian_matrix), which have the 
property that $(QH)^T = QH$ and are commonly used in physics.

```@autodocs
Modules = [Manifolds]
Pages = ["manifolds/Symplectic.jl"]
Order = [:type, :function]
```

## Literature
