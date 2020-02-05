# Notation overview

Since manifolds include a reasonable amount of elements and functions, the following list tries to keep an overview of used notation throughout `Manifolds.jl`.
The order is alphabetically by name.
They might be used in a plain form within the code or when referring to that code.
This is for example the case the calligraphic symbols.

Within the documented functions the utf8 symbols are used whenever possible,
as long as that renders still in $\TeX$ within this documentation.

| Symbol | Description | Also used | Comment |
|:--:|:--------------- |:--:|:-- |
| $\times$ | Cartesian product of two manifolds | | see [`ProductManifold](@ref) |
| $\^$ | (n-ary) Cartesian power | | see [`PowerManifold`](@ref) |
| $T^*_p \mathcal M$ | the cotangent space at $p$ | | |
| $\xi$ | a cotangent vector from $T^*_p \mathcal M$ | $\xi_1, \xi_2,\ldots,\eta,\zeta$ | sometimes written with base point $\xi_p$. |
| $n$ | dimension (of a manifold) | $n_1,n_2,\ldots,m, \operatorname{dim}(\mathcal M)$| for the real dimension sometimes also $\operatorname{dim}_{\mathbb R}(\mathcal M)$|
| $d(\cdot,\cdot)$ | (Riemannian) distance | $d_{\mathcal M}(\cdot,\cdot)$ | |
| $F$ | a fiber | | |
| $\mathbb F$ | a field | | field a manifold is based on, usually $\mathcal F \in \{\mathbb R,\mathbb C\}$ |
| $\gamma$ | a geodesic | $\gamma_{p;q}$, $\gamma_{p,X}$ | connecting two points $p,q$ or starting in $p$ with velocity $X$. |
| $\circ$ | a group operation | |
| $e$ | identity element of a group | |
| $k$ | indices | $i,j$ | |
| $\langle\cdot,\cdot\rangle$ | inner product (in $T_p \mathcal M$) | $\langle\cdot,\cdot\rangle_p, g_p(\cdot,\cdot)$ |
| $\mathfrak g$ | a Lie algebra | |
| $\mathcal{G}$ | a (Lie) group | |
| $\mathcal M$ | a manifold | $\mathcal M_1, \mathcal M_2,\ldots,\mathcal N$ | |
| $\operatorname{Exp}$ | the matrix exponential | |
| $\operatorname{Log}$ | the matrix logarithm | |
| $\mathcal P_{q\gets p}X$ | parallel transport | | of the vector $X$ from $T_p\mathcal M$ to $T_q\mathcal M$
| $p$ | a point on $\mathcal M$ | $p_1, p_2, \ldots,q$ | for 3 points one might use $x,y,z$ |
| $T_p \mathcal M$ | the tangent space at $p$ | | |
| $X$ | a tangent vector from $T_p \mathcal M$ | $X_1,X_2,\ldots,Y,Z$ | sometimes written with base point $X_p$ |
| $B$ | a vector bundle | |
