raw"""
    submersion(M::AbstractManifold, q)

Evaluate the submersion function in the embedding of `M` at `q`.

Let ``\mathcal M` be an embedded manifold in some manifold ``\mathcal N``.

This function then evaluates the (local) defining function (or submersion)
[Boumal:2023; Def. 3.10](@cite)

```math
  c: \mathcal N → ℝ^k \quad\text{ with } c(q) = 0 ⇔ q ∈ \mathcal M,
```

where the differential of ``c``, ``Dc`` has rank ``k`` everywhere.

!!! info "Naming

   This is a special case of the submersion theorem [AbsilMahonySepulchre:2008; Prop. 3.3.3](@cite)
   with ``\mathcal M_1 = \mathcal N`` and ``\mathcal M_2 = ℝ^k``.

!!! note

    This function only makes sense, if the dimension of ``\mathcal N`` is larger than the one of ``\mathcal M``.
    If both are of equal dimension, ``\mathcal M`` is an open subset of ``\mathcal N``.

# See also
[`differential_submersion`](@ref)
"""
function get_submersion(M::AbstractManifold, q) end

raw"""
    differential_submersion(M, q)
    differential_submersion(M, q, Y)

Evaluate the derivative of the submersion of ``\mathcal M`` with respect to ``q ∈ \mathcal N``.

Let ``\mathcal M` be an embedded manifold in some manifold ``\mathcal N``.

This function then evaluates the (local) defining function (or submersion)
[Boumal:2023; Def. 3.10](@cite)

```math
c: \mathcal N → ℝ^k \quad\text{ with } c(q) = 0 ⇔ q ∈ \mathcal M.
```

It's differential denoted by ``Dc(q): T_q\mathcal N → ℝ^{k}``, and we have that at a point
``p ∈ \mathcal M`` we get ``Dc(p)[X] = 0 ⇔ X ∈ T_p\mathcal M``.

The first signature returns a matrix representation of the differential, or in other words
the second can then be computed by ``D(q)*Y``, but there might exist more efficient versions for the second.

# See also
[`[`submersion`](@ref)
"""
function differential_submersion(M::AbstractManifold, q, args...) end
