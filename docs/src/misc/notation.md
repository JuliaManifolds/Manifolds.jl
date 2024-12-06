# Notation overview

Since manifolds include a reasonable amount of elements and functions, the following list tries to keep an overview of used notation throughout `Manifolds.jl`.
The order is alphabetical by name.
They might be used in a plain form within the code or when referring to that code.
This is for example the case with the calligraphic symbols.

Within the documented functions, the utf8 symbols are used whenever possible, as long as that renders correctly in ``\TeX`` within this documentation.

| Symbol | Description | Also used | Comment |
|:--:|:--------------- |:--:|:-- |
| ``\tau_p`` | action map by group element ``p`` | ``\mathrm{L}_p``, ``\mathrm{R}_p`` | either left or right |
| ``\operatorname{Ad}_p(X)`` | adjoint action of element ``p`` of a Lie group on the element ``X`` of the corresponding Lie algebra | | |
| ``×`` | Cartesian product of two manifolds | | see [`ProductManifold`](@extref `ManifoldsBase.ProductManifold`) |
| ``^{\wedge}`` | (n-ary) Cartesian power of a manifold | | see [`PowerManifold`](@extref `ManifoldsBase.PowerManifold`) |
| ``⋅^\mathrm{H}`` | conjugate/Hermitian transpose | | |
| ``a`` | coordinates of a point in a chart | | see [`get_parameters`](@ref) |
| ``\frac{\mathrm{D}}{\mathrm{d}t}`` | covariant derivative of a vector field ``X(t)`` | | |
| ``T^*_p \mathcal M`` | the cotangent space at ``p`` | | |
| ``ξ`` | a cotangent vector from ``T^*_p \mathcal M`` | ``ξ_1, ξ_2,… ,η,\zeta`` | sometimes written with base point ``ξ_p``. |
| ``\mathrm{d}\phi_p(q)`` | Differential of a map ``\phi: \mathcal M → \mathcal N`` with respect to ``p`` at a point ``q``. For functions of multiple variables, for example ``\phi(p, p_1)`` where ``p \in \mathcal M`` and ``p_1 \in \mathcal M_1``, variable ``p`` is explicitly stated to specify with respect to which argument the differential is calculated. | ``\mathrm{d}\phi_q``, ``(\mathrm{d}\phi)_q``, ``(\phi_*)_q``, ``D_p\phi(q)`` | pushes tangent vectors ``X \in T_q \mathcal M`` forward to ``\mathrm{d}\phi_p(q)[X] \in T_{\phi(q)} \mathcal N`` |
| ``n`` | dimension (of a manifold) | ``n_1,n_2,\ldots,m, \dim(\mathcal M)``| for the real dimension sometimes also ``\dim_{\mathbb R}(\mathcal M)``|
| ``d(⋅,⋅)`` | (Riemannian) distance | ``d_{\mathcal M}(⋅,⋅)`` | |
| ``\exp_p X`` | exponential map at ``p \in \mathcal M`` of a vector ``X \in T_p \mathcal M`` | ``\exp_p(X)`` | |
| ``F`` | a fiber | | see [`Fiber`](@extref `ManifoldsBase.Fiber`) |
| ``\mathbb F`` | a field, usually ``\mathbb F \in \{\mathbb R,\mathbb C, \mathbb H\}``, i.e. the real, complex, and quaternion numbers, respectively. | |field a manifold or a basis is based on |
| ``\gamma`` | a geodesic | ``\gamma_{p;q}``, ``\gamma_{p,X}`` | connecting two points ``p,q`` or starting in ``p`` with velocity ``X``. |
| ``\operatorname{grad} f(p)`` | (Riemannian) gradient of function ``f \colon \mathcal{M} → ℝ`` at ``p \in \mathcal{M}`` | | |
| ``\nabla f(p)`` | (Euclidean) gradient of function ``f \colon \mathcal{M} → ℝ`` at ``p \in \mathcal{M}`` but thought of as evaluated in the embedding | `G` | |
| ``\circ`` | a group operation | | |
| ``⋅^\mathrm{H}`` | Hermitian or conjugate transposed for both complex or quaternion matrices| | |
| ``\operatorname{Hess} f(p)`` | (Riemannian) Hessian of function ``f \colon T_p\mathcal{M} → T_p\mathcal M`` (i.e. the 1-1-tensor form) at ``p \in \mathcal{M}`` | | |
| ``\nabla^2 f(p)`` | (Euclidean) Hessian of function ``f`` in the embedding | `H` | |
| ``e`` | identity element of a group | | |
| ``I_k`` | identity matrix of size ``k×k`` | | |
| ``k`` | indices | ``i,j`` | |
| ``\langle⋅,⋅\rangle`` | inner product (in ``T_p \mathcal M``) | ``\langle⋅,⋅\rangle_p, g_p(⋅,⋅)`` | |
| ``\operatorname{retr}^{-1}_pq``|  an inverse retraction | | |
| ``\mathfrak g`` | a Lie algebra | | |
| ``\mathcal{G}`` | a (Lie) group | | |
| ``\log_p q`` | logarithmic map at ``p \in \mathcal M`` of a point ``q \in \mathcal M`` | ``\log_p(q)`` | |
| ``\mathcal M`` | a manifold | ``\mathcal M_1, \mathcal M_2,\ldots,\mathcal N`` | |
| ``N_p \mathcal M`` | the normal space of the tangent space ``T_p \mathcal M`` in some embedding ``\mathcal E`` that should be clear from context | | |
| ``V`` | a normal vector from ``N_p \mathcal M`` | ``W`` | |
| ``\operatorname{Exp}`` | the matrix exponential | | |
| ``\operatorname{Log}`` | the matrix logarithm | | |
| ``\mathcal P_{q\gets p}X`` | parallel transport | | of the vector ``X`` from ``T_p\mathcal M`` to ``T_q\mathcal M`` |
| ``\mathcal P_{p,Y}X`` | parallel transport in direction ``Y`` | | of the vector ``X`` from ``T_p\mathcal M`` to ``T_q\mathcal M``, ``q = \exp_pY`` |
| ``\mathcal P_{t_1\gets t_0}^cX`` | parallel transport along the curve ``c``| ``\mathcal P^cX=\mathcal P_{1\gets 0}^cX`` | of the vector ``X`` from ``p=c(0)`` to ``c(1)`` |
| ``p`` | a point on ``\mathcal M`` | ``p_1, p_2, \ldots,q`` | for 3 points one might use ``x,y,z`` |
| ``\operatorname{retr}_pX``|  a retraction | | |
| ``\kappa_p(X, Y)`` | sectional curvature | | |
| ``ξ`` | a set of tangent vectors | ``\{X_1,\ldots,X_n\}`` | |
| ``J_{2n} \in ℝ^{2n×2n}`` | the [`SymplecticElement`](@ref) | | |
| ``T_p \mathcal M`` | the tangent space at ``p`` | | |
| ``X`` | a tangent vector from ``T_p \mathcal M`` | ``X_1,X_2,\ldots,Y,Z`` | sometimes written with base point ``X_p`` |
| ``\operatorname{tr}`` | trace (of a matrix) | | |
| ``⋅^\mathrm{T}`` | transposed | | |
| ``e_i \in \mathbb R^n`` | the ``i``th unit vector | ``e_i^n`` | the space dimension (``n``) is omitted, when clear from context |
| ``B`` | a vector bundle | | |
| ``\mathcal T_{q\gets p}X`` | vector transport | | of the vector ``X`` from ``T_p\mathcal M`` to ``T_q\mathcal M`` |
| ``\mathcal T_{p,Y}X`` | vector transport in direction ``Y`` | | of the vector ``X`` from ``T_p\mathcal M`` to ``T_q\mathcal M``, where ``q`` is determined by ``Y``, for example using the exponential map or some retraction. |
| ``\operatorname{Vol}(\mathcal M)`` | volume of manifold ``\mathcal M`` | | |
| ``\theta_p(X)`` | volume density for vector ``X`` tangent at point ``p`` | | |
| ``\mathcal W`` | the Weingarten map ``\mathcal W: T_p\mathcal M × N_p\mathcal M → T_p\mathcal M`` | ``\mathcal W_p`` | the second notation to emphasize the dependency of the point ``p\in\mathcal M`` |
| ``0_k`` | the ``k×k`` zero matrix. | | |

## Comparison with notation commonly used in robotics

In robotics, a different notation is commonly used.
The table below shows a quick guide how to translate between them for people coming from robotics background.
We use [SolaDerayAtchuthan:2021](@cite) as the primary robotics source.

| Robotics concept | Manifolds.jl notation |
|:--|:--------------- |
| ``p \circ q`` | `compose(G, p, q)` |
| ``p^{-1}``|  `inv(G, p)` |
| ``\mathcal{E}`` | `Identity(G)` or `identity_element(G)` |
| group action ``p\cdot p_m`` | `apply(A, p, p_m)` |
| Lie group exponential ``\exp\colon \mathfrak{g} \to \mathcal{G}``, ``\exp(X)=p`` | `exp_lie(G, p)` |
| Lie group logarithm ``\log\colon \mathcal{G} \to \mathfrak{g}``, ``\log(p)=X`` | `log_lie(G, X)` |
| ``n``-D vector | `TranslationGroup(n)`;  its action is `TranslationAction(Euclidean(n), TranslationGroup(n))` |
| circle ``S^1`` | `CircleGroup()`; its action is `ComplexPlanarRotation` |
| rotation ``\mathrm{SO}(n)`` | `SpecialOrthogonal(n)`; its action is `RotationAction(Euclidean(n), SpecialOrthogonal(n))` |
| rigid motion ``\mathrm{SE}(n)`` | `SpecialEuclidean(n)`; its action is `RotationTranslationAction(Euclidean(n), SpecialEuclidean(n))` |
| unit quaternions ``S^3`` | `UnitaryMatrices(1, H)`; note that 3-sphere and the group of rotations (with its bi-invariant metric) are homeomorphic but not isomorphic |
| size (as in Table I) | related to `representation_size(G)` |
| dim (as in Table I) | `manifold_dimension(G)` |
| Lie algebra element with coordinates ``\tau^{\wedge}`` | `hat(G, Identity(G), tau)` |
| coordinates of an element of Lie algebra ``X^{\vee}`` | `vee(G, Identity(G), X)` |
| capital exponential map ``\operatorname{Exp}`` | `exp_lie(G, hat(G, Identity(G), tau))` |
| capital logarithmic map ``\operatorname{Log}`` | `vee(G, Identity(G), log_lie(G, p))` |
| right-``\oplus``, ``p \oplus \tau`` | `compose(G, exp_lie(G, hat(G, Identity(G), tau)))` |
| right-``\ominus``, ``p \ominus q`` | `vee(G, Identity(G), log_lie(G, compose(G, inv(G, q), p)))`|
| left-``\oplus``, ``\tau \oplus p`` | `compose(G, exp_lie(G, hat(G, Identity(G), tau)), p)` |
| left-``\ominus``, ``p \ominus q`` | `vee(G, Identity(G), log_lie(G, compose(G, p, inv(G, q))))` |
| adjoint ``\mathrm{Ad}_{p}(\tau^{\wedge})`` | `adjoint_action(G, p, hat(G, Identity(G), tau))` |
| adjoint matrix ``\mathrm{Ad}_{p}`` | `adjoint_matrix(G, p)` |
| Jacobian of group inversion and composition | these can be easily constructed from the adjoint matrix |
| left and right Jacobians of a function | In JuliaManifolds there is always one preferred way to store tangent vectors specified by each manifold, and so we follow the standard mathematical convention of having one Jacobian which follows the selected tangent vector storage convention. See for example `jacobian_exp_argument`, `jacobian_exp_basepoint`, `jacobian_log_argument`, `jacobian_log_basepoint` from `ManifoldDiff.jl`. |
| left and right Jacobians (of a group) ``\mathbf{J}_l, \mathbf{J}_r`` | `jacobian_exp_argument` for exponential coordinates. For other coordinate systems no replacement is available yet. |
| Jacobians of group actions | not available yet |

Be also careful that the meaning of ``\mathbf{x}`` is inconsistent in Table I from [SolaDerayAtchuthan:2021](@cite). It's a complex number for circle, quaternion for quaternion rotation and column vectors for other rows.
