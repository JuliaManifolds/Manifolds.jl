# Notation overview

Since manifolds include a reasonable amount of elements and functions, the following list tries to keep an overview of used notation throughout `Manifolds.jl`.
The order is alphabetical by name.
They might be used in a plain form within the code or when referring to that code.
This is for example the case with the calligraphic symbols.

Within the documented functions, the utf8 symbols are used whenever possible, as long as that renders correctly in ``\TeX`` within this documentation.

| Symbol | Description | Also used | Comment |
|:--:|:--------------- |:--:|:-- |
| ``\tau_p`` | action map by group element ``p`` | ``\mathrm{L}_p``, ``\mathrm{R}_p`` | either left or right |
| ``\times`` | Cartesian product of two manifolds | | see [`ProductManifold`](@ref) |
| ``^{\wedge}`` | (n-ary) Cartesian power of a manifold | | see [`PowerManifold`](@ref) |
| ``T^*_p \mathcal M`` | the cotangent space at ``p`` | | |
| ``\xi`` | a cotangent vector from ``T^*_p \mathcal M`` | ``\xi_1, \xi_2,\ldots,\eta,\zeta`` | sometimes written with base point ``\xi_p``. |
| ``\mathrm{d}\phi_p(q)`` | Differential of a map ``\phi: \mathcal M \to \mathcal N`` with respect to ``p`` at a point ``q``. For functions of multiple variables, for example ``\phi(p, p_1)`` where ``p \in \mathcal M`` and ``p_1 \in \mathcal M_1``, variable ``p`` is explicitly stated to specify with respect to which argument the differential is calculated. | ``\mathrm{d}\phi_q``, ``(\mathrm{d}\phi)_q``, ``(\phi_*)_q``, ``D_p\phi(q)`` | pushes tangent vectors ``X \in T_q \mathcal M`` forward to ``\mathrm{d}\phi_p(q)[X] \in T_{\phi(q)} \mathcal N`` |
| ``n`` | dimension (of a manifold) | ``n_1,n_2,\ldots,m, \dim(\mathcal M)``| for the real dimension sometimes also ``\dim_{\mathbb R}(\mathcal M)``|
| ``d(\cdot,\cdot)`` | (Riemannian) distance | ``d_{\mathcal M}(\cdot,\cdot)`` | |
| ``\exp_p X`` | exponential map at ``p \in \mathcal M`` of a vector ``X \in T_p \mathcal M`` | ``\exp_p(X)`` | |
| ``F`` | a fiber | | see [`VectorBundleFibers`](@ref) |
| ``\mathbb F`` | a field, usually ``\mathbb F \in \{\mathbb R,\mathbb C, \mathbb H\}``, i.e. the real, complex, and quaternion numbers, respectively. | |field a manifold or a basis is based on |
| ``\gamma`` | a geodesic | ``\gamma_{p;q}``, ``\gamma_{p,X}`` | connecting two points ``p,q`` or starting in ``p`` with velocity ``X``. |
| ``\nabla f(p)`` | gradient of function ``f \colon \mathcal{M} \to \mathbb{R}`` at ``p \in \mathcal{M}`` | | |
| ``\circ`` | a group operation | |
| ``\cdot^\mathrm{H}`` | Hermitian or conjugate transposed| |
| ``e`` | identity element of a group | |
| ``I_k`` | identity matrix of size ``k\times k`` | |
| ``k`` | indices | ``i,j`` | |
| ``\langle\cdot,\cdot\rangle`` | inner product (in ``T_p \mathcal M``) | ``\langle\cdot,\cdot\rangle_p, g_p(\cdot,\cdot)`` |
| ``\operatorname{retr}^{-1}_pq``|  an inverse retraction | |
| ``\mathfrak g`` | a Lie algebra | |
| ``\mathcal{G}`` | a (Lie) group | |
| ``\log_p q`` | logarithmic map at ``p \in \mathcal M`` of a point ``q \in \mathcal M`` | ``\log_p(q)`` | |
| ``\mathcal M`` | a manifold | ``\mathcal M_1, \mathcal M_2,\ldots,\mathcal N`` | |
| ``\operatorname{Exp}`` | the matrix exponential | |
| ``\operatorname{Log}`` | the matrix logarithm | |
| ``\mathcal P_{q\gets p}X`` | parallel transport | | of the vector ``X`` from ``T_p\mathcal M`` to ``T_q\mathcal M``
| ``\mathcal P_{p,Y}X`` | parallel transport in direction ``Y`` | | of the vector ``X`` from ``T_p\mathcal M`` to ``T_q\mathcal M``, ``q = exp_pY``
| ``p`` | a point on ``\mathcal M`` | ``p_1, p_2, \ldots,q`` | for 3 points one might use ``x,y,z`` |
| ``\operatorname{retr}_pX``|  a retraction | |
| ``\Xi`` | a set of tangent vectors | ``\{X_1,\ldots,X_n\}`` | |
| ``T_p \mathcal M`` | the tangent space at ``p`` | | |
| ``X`` | a tangent vector from ``T_p \mathcal M`` | ``X_1,X_2,\ldots,Y,Z`` | sometimes written with base point ``X_p`` |
| ``\operatorname{tr}`` | trace (of a matrix) | |
| ``\cdot^\mathrm{T}`` | transposed | |
| ``B`` | a vector bundle | |
| ``\mathcal T_{q\gets p}X`` | vector transport | | of the vector ``X`` from ``T_p\mathcal M`` to ``T_q\mathcal M``
| ``\mathcal T_{p,Y}X`` | vector transport in direction ``Y`` | | of the vector ``X`` from ``T_p\mathcal M`` to ``T_q\mathcal M``, where ``q`` is deretmined by ``Y``, for example using the exponential map or some retraction.
| ``0_k`` | the ``k\times k`` zero matrix. | |
