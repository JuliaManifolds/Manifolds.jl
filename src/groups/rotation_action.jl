@doc raw"""
    RotationAction(
        M::AbstractManifold,
        SOn::SpecialOrthogonal,
        AD::ActionDirection = LeftAction(),
    )

Space of actions of the [`SpecialOrthogonal`](@ref) group $\mathrm{SO}(n)$ on a
Euclidean-like manifold `M` of dimension `n`.
"""
struct RotationAction{TAD <: ActionDirection, TM <: AbstractManifold, TSO <: SpecialOrthogonal} <:
    AbstractGroupAction{TAD}
    manifold::TM
    SOn::TSO
end

function RotationAction(
        M::AbstractManifold,
        SOn::SpecialOrthogonal,
        ::TAD = LeftAction(),
    ) where {TAD <: ActionDirection}
    return RotationAction{TAD, typeof(M), typeof(SOn)}(M, SOn)
end

function Base.show(io::IO, A::RotationAction)
    return print(io, "RotationAction($(A.manifold), $(A.SOn), $(direction(A)))")
end

const RotationActionOnVector{TAD, 𝔽, TE, TSO} = RotationAction{
    TAD,
    <:Union{Euclidean{TE, 𝔽}, TranslationGroup{TE, 𝔽}},
    SpecialOrthogonal{TSO},
} where {TAD <: ActionDirection, 𝔽, TE, TSO}

base_group(A::RotationAction) = A.SOn

group_manifold(A::RotationAction) = A.manifold

function switch_direction(A::RotationAction{TAD}) where {TAD <: ActionDirection}
    return RotationAction(A.manifold, A.SOn, switch_direction(TAD()))
end

apply(::RotationActionOnVector{LeftAction}, a, p) = a * p
function apply(A::RotationActionOnVector{RightAction}, a, p)
    return inv(base_group(A), a) * p
end

apply!(::RotationActionOnVector{LeftAction}, q, a, p) = mul!(q, a, p)

function inverse_apply(A::RotationActionOnVector{LeftAction}, a, p)
    return inv(base_group(A), a) * p
end
inverse_apply(::RotationActionOnVector{RightAction}, a, p) = a * p

apply_diff(::RotationActionOnVector{LeftAction}, a, p, X) = a * X
function apply_diff(
        ::RotationActionOnVector{LeftAction},
        ::Identity{MultiplicationOperation},
        p,
        X,
    )
    return X
end
function apply_diff(A::RotationActionOnVector{RightAction}, a, p, X)
    return inv(base_group(A), a) * X
end

function apply_diff!(::RotationActionOnVector{LeftAction}, Y, a, p, X)
    return mul!(Y, a, X)
end
function apply_diff!(A::RotationActionOnVector{RightAction}, Y, a, p, X)
    return mul!(Y, inv(base_group(A), a), X)
end

function apply_diff_group(::RotationActionOnVector{LeftAction}, ::Identity, X, p)
    return X * p
end

function apply_diff_group!(::RotationActionOnVector{LeftAction}, Y, ::Identity, X, p)
    Y .= X * p
    return Y
end

function inverse_apply_diff(A::RotationActionOnVector{LeftAction}, a, p, X)
    return inv(base_group(A), a) * X
end
function inverse_apply_diff(::RotationActionOnVector{RightAction}, a, p, X)
    return a * X
end

function optimal_alignment(::RotationActionOnVector{LeftAction}, p, q)
    Xmul = p * transpose(q)
    F = svd(Xmul)
    L = size(Xmul)[2]
    UVt = F.U * F.Vt
    Ostar = det(UVt) ≥ 0 ? UVt : F.U * Diagonal([i < L ? 1 : -1 for i in 1:L]) * F.Vt
    return convert(typeof(Xmul), Ostar)
end
function optimal_alignment(A::RotationActionOnVector{RightAction}, p, q)
    return optimal_alignment(switch_direction(A), q, p)
end

@doc raw"""
    RotationAroundAxisAction(axis::AbstractVector)

Space of actions of the circle group [`RealCircleGroup`](@ref) on $ℝ^3$ around given `axis`.
"""
struct RotationAroundAxisAction{TA <: AbstractVector} <: AbstractGroupAction{LeftAction}
    axis::TA
end

base_group(::RotationAroundAxisAction) = RealCircleGroup()

group_manifold(::RotationAroundAxisAction) = Euclidean(3)

@doc raw"""
    apply(A::RotationAroundAxisAction, θ, p)

Rotate point `p` from [`Euclidean(3)`](@ref) manifold around axis `A.axis` by angle `θ`.
The formula reads
````math
p_{rot} = (\cos(θ))p + (k×p) \sin(θ) + k (k⋅p) (1-\cos(θ)),
````
where ``k`` is the vector `A.axis` and `⋅` is the dot product.
"""
function apply(A::RotationAroundAxisAction, θ, p)
    sθ, cθ = sincos(θ)
    apd = dot(A.axis, p)
    return p .* cθ .+ cross(A.axis, p) .* sθ .+ A.axis .* apd .* (1 - cθ)
end
apply(::RotationAroundAxisAction, ::Identity{AdditionOperation}, p) = p
function apply(A::RotationAroundAxisAction, θ::AbstractArray, p)
    # this method is here to make sure that θ represented by 1-element vectors works
    return apply(A, θ[], p)
end

function apply!(A::RotationAroundAxisAction, q, θ, p)
    return copyto!(q, apply(A, θ, p))
end

function inverse_apply(A::RotationAroundAxisAction, θ, p)
    return apply(A, -θ, p)
end

###

@doc raw"""
    RowwiseMultiplicationAction{
        TAD<:ActionDirection,
        TM<:AbstractManifold,
        TO<:GeneralUnitaryMultiplicationGroup,
    } <: AbstractGroupAction{TAD}

Action of the (special) unitary or orthogonal group [`GeneralUnitaryMultiplicationGroup`](@ref)
of type `On` columns of points on a matrix manifold `M`.

# Constructor

    RowwiseMultiplicationAction(
        M::AbstractManifold,
        On::GeneralUnitaryMultiplicationGroup,
        AD::ActionDirection = LeftAction(),
    )
"""
struct RowwiseMultiplicationAction{
        TAD <: ActionDirection,
        TM <: AbstractManifold,
        TO <: GeneralUnitaryMultiplicationGroup,
    } <: AbstractGroupAction{TAD}
    manifold::TM
    On::TO
end

function RowwiseMultiplicationAction(
        M::AbstractManifold,
        On::GeneralUnitaryMultiplicationGroup,
        ::TAD = LeftAction(),
    ) where {TAD <: ActionDirection}
    return RowwiseMultiplicationAction{TAD, typeof(M), typeof(On)}(M, On)
end

const LeftRowwiseMultiplicationAction{
    TM <: AbstractManifold,
    TO <: GeneralUnitaryMultiplicationGroup,
} = RowwiseMultiplicationAction{LeftAction, TM, TO}

function apply(::LeftRowwiseMultiplicationAction, a, p)
    return (a * p')'
end
function apply(::LeftRowwiseMultiplicationAction, ::Identity{MultiplicationOperation}, p)
    return p
end

function apply!(::LeftRowwiseMultiplicationAction, q, a, p)
    return map((qrow, prow) -> mul!(qrow, a, prow), eachrow(q), eachrow(p))
end

base_group(A::RowwiseMultiplicationAction) = A.On

group_manifold(A::RowwiseMultiplicationAction) = A.manifold

function inverse_apply(::LeftRowwiseMultiplicationAction, a, p)
    return (a \ p')'
end

###

const MatrixGroup{T, 𝔽} = Union{
    GeneralUnitaryMultiplicationGroup{T, 𝔽},
    GeneralLinear{T, 𝔽},
    SpecialLinear{T, 𝔽},
} where {T, 𝔽}

@doc raw"""
    ColumnwiseMultiplicationAction{
        TAD<:ActionDirection,
        TM<:AbstractManifold,
        TO<:MatrixGroup,
    } <: AbstractGroupAction{TAD}

Action of the (special) unitary or orthogonal group [`GeneralUnitaryMultiplicationGroup`](@ref)
or [`GeneralLinear`](@ref) group
of type `On` columns of points on a matrix manifold `M`.

# Constructor

    ColumnwiseMultiplicationAction(
        M::AbstractManifold,
        On::MatrixGroup,
        AD::ActionDirection = LeftAction(),
    )
"""
struct ColumnwiseMultiplicationAction{
        TAD <: ActionDirection,
        TM <: AbstractManifold,
        TO <: MatrixGroup,
    } <: AbstractGroupAction{TAD}
    manifold::TM
    On::TO
end

function ColumnwiseMultiplicationAction(
        M::AbstractManifold,
        On::MatrixGroup,
        ::TAD = LeftAction(),
    ) where {TAD <: ActionDirection}
    return ColumnwiseMultiplicationAction{TAD, typeof(M), typeof(On)}(M, On)
end

const LeftColumnwiseMultiplicationAction{TM <: AbstractManifold, TO <: MatrixGroup} =
    ColumnwiseMultiplicationAction{LeftAction, TM, TO}

function apply(::LeftColumnwiseMultiplicationAction, a, p)
    return a * p
end
function apply(::LeftColumnwiseMultiplicationAction, ::Identity{MultiplicationOperation}, p)
    return p
end

function apply!(::LeftColumnwiseMultiplicationAction, q, a, p)
    return map((qrow, prow) -> mul!(qrow, a, prow), eachcol(q), eachcol(p))
end

base_group(A::LeftColumnwiseMultiplicationAction) = A.On

group_manifold(A::LeftColumnwiseMultiplicationAction) = A.manifold

function inverse_apply(::LeftColumnwiseMultiplicationAction, a, p)
    return a \ p
end

@doc raw"""
    optimal_alignment(A::LeftColumnwiseMultiplicationAction, p, q)

Compute optimal alignment for the left [`ColumnwiseMultiplicationAction`](@ref), i.e. the
group element ``O^{*}`` that, when it acts on `p`, returns the point closest to `q`. Details
of computation are described in Section 2.2.1 of [SrivastavaKlassen:2016](@cite).

The formula reads
```math
O^{*} = \begin{cases}
UV^T & \text{if } \operatorname{det}(p q^{\mathrm{T}}) \geq 0\\
U K V^{\mathrm{T}} & \text{otherwise}
\end{cases}
```
where ``U \Sigma V^{\mathrm{T}}`` is the SVD decomposition of ``p q^{\mathrm{T}}`` and ``K``
is the unit diagonal matrix with the last element on the diagonal replaced with -1.
"""
function optimal_alignment(
        A::LeftColumnwiseMultiplicationAction{
            <:AbstractManifold,
            <:GeneralUnitaryMultiplicationGroup,
        },
        p,
        q,
    )
    is_point(A.manifold, p; error = :error)
    is_point(A.manifold, q; error = :error)

    Xmul = p * transpose(q)
    F = svd(Xmul)
    L = size(Xmul)[2]
    UVt = F.U * F.Vt
    Ostar = det(UVt) ≥ 0 ? UVt : F.U * Diagonal([i < L ? 1 : -1 for i in 1:L]) * F.Vt
    return convert(typeof(Xmul), Ostar)
end

@doc raw"""
    ComplexPlanarRotation()

Action of the circle group [`CircleGroup`](@ref) on ``ℝ^2`` by left multiplication.
"""
struct ComplexPlanarRotation <: AbstractGroupAction{LeftAction} end

base_group(::ComplexPlanarRotation) = CircleGroup()

group_manifold(::ComplexPlanarRotation) = Euclidean(2)

@doc raw"""
    apply(A::ComplexPlanarRotation, g::Complex, p)

Rotate point `p` from [`Euclidean(2)`](@ref) manifold by the group element `g`.
The formula reads
````math
p_{rot} =  \begin{bmatrix}
\cos(θ) & \sin(θ)\\
-\sin(θ) & \cos(θ)
\end{bmatrix} p,
````
where `θ` is the argument of complex number `g`.
"""
function apply(::ComplexPlanarRotation, g::Complex, p)
    sinθ, cosθ = g.im, g.re
    return (@SMatrix [cosθ -sinθ; sinθ cosθ]) * p
end
apply(::ComplexPlanarRotation, ::Identity{MultiplicationOperation}, p) = p

function apply!(A::ComplexPlanarRotation, q, g::Complex, p)
    return copyto!(q, apply(A, g, p))
end
function apply!(::ComplexPlanarRotation, q, ::Identity{MultiplicationOperation}, p)
    return copyto!(q, p)
end

@doc raw"""
    QuaternionRotation

Action of the unit quaternion group [`Unitary`](@ref)`(1, ℍ)` on ``ℝ^3``.
"""
struct QuaternionRotation <: AbstractGroupAction{LeftAction} end

base_group(::QuaternionRotation) = Unitary(1, ℍ)

group_manifold(::QuaternionRotation) = Euclidean(3)

@doc raw"""
    apply(A::QuaternionRotation, g::Quaternion, p)

Rotate point `p` from [`Euclidean`](@ref)`(3)` manifold through conjugation by the group
element `g`.
The formula reads
````math
(0, p_{rot,x}, p_{rot,y}, p_{rot,z}) = g ⋅ (0, p_x, p_y, p_z) ⋅ g^{\mathrm{H}}
````
where ``(0, p_x, p_y, p_z)`` is quaternion with non-real coefficients from encoding
the point `p` and ``g^{\mathrm{H}}`` is quaternion conjugate of ``g``.
"""
function apply(::QuaternionRotation, g::Quaternions.Quaternion, p::SVector)
    p_quat = Quaternions.Quaternion(0, p[1], p[2], p[3])
    p_conj = g * p_quat * conj(g)
    return @SVector [p_conj.v1, p_conj.v2, p_conj.v3]
end
apply(::QuaternionRotation, ::Identity{MultiplicationOperation}, p) = p

function apply!(::QuaternionRotation, q, g::Quaternions.Quaternion, p)
    p_quat = Quaternions.Quaternion(0, p[1], p[2], p[3])
    p_conj = g * p_quat * conj(g)
    q[1] = p_conj.v1
    q[2] = p_conj.v2
    q[3] = p_conj.v3
    return q
end
function apply!(::QuaternionRotation, q, ::Identity{MultiplicationOperation}, p)
    return copyto!(q, p)
end

"""
    quaternion_rotation_matrix(g::Quaternions.Quaternion)

Compute rotation matrix for [`RotationAction`](@ref) corresponding to
[`QuaternionRotation`](@ref) by `g`.

See https://www.songho.ca/opengl/gl_quaternion.html for details.
"""
function quaternion_rotation_matrix(g::Quaternions.Quaternion)
    r11 = 1 - 2 * (g.v2^2 + g.v3^2)
    r12 = 2 * (g.v1 * g.v2 - g.v3 * g.s)
    r13 = 2 * (g.v1 * g.v3 + g.v2 * g.s)
    r21 = 2 * (g.v1 * g.v2 + g.v3 * g.s)
    r22 = 1 - 2 * (g.v1^2 + g.v3^2)
    r23 = 2 * (g.v2 * g.v3 - g.v1 * g.s)
    r31 = 2 * (g.v1 * g.v3 - g.v2 * g.s)
    r32 = 2 * (g.v2 * g.v3 + g.v1 * g.s)
    r33 = 1 - 2 * (g.v1^2 + g.v2^2)

    return @SMatrix [
        r11 r12 r13
        r21 r22 r23
        r31 r32 r33
    ]
end
