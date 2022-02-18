# Symplectic Stiefel

The [`SymplecticStiefel`](@ref) manifold, denoted $\operatorname{SpSt}(2n, 2k)$, 
represents canonical symplectic bases of $2k$ dimensonal symplectic subspaces of $\mathbb{R}^{2n \times 2n}$. 
This means that the columns of each element $p \in \operatorname{SpSt}(2n, 2k) \subset \mathbb{R}^{2n \times 2k}$ 
constitute a canonical symplectic basis of $\operatorname{span}(p)$. 
The canonical symplectic form is a non-degenerate, bilinear, and skew symmetric map 
$\omega_{2k}\colon \mathbb{F}^{2k} \times \mathbb{F}^{2k} 
\rightarrow \mathbb{F}$, given by
$\omega_{2k}(x, y) = x^T Q_{2k} y$ for elements $x, y \in \mathbb{F}^{2k}$, with
````math
    Q_{2k} = 
    \begin{bmatrix}
     0_k  &  I_k \\
    -I_k  &  0_k
    \end{bmatrix}.
```` 
Specifically given an element $p \in \operatorname{SpSt}(2n, 2k)$ we require that
````math
    \omega_{2n} (p x, p y) = x^T(p^TQ_{2n}p)y = x^TQ_{2k}y = \omega_{2k}(x, y) \;\forall\; x, y \in \mathbb{F}^{2k},
````
leading to the requirement on $p$ that $p^TQ_{2n}p = Q_{2k}$.
In the case that $k = n$, this manifold reduces to the [`Symplectic`](@ref) manifold, which is also known as the symplectic group.

```@autodocs
Modules = [Manifolds]
Pages = ["manifolds/SymplecticStiefel.jl"]
Order = [:type, :function]
```

## Literature
