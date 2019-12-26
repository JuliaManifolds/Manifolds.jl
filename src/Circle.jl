@doc doc"""
    Circle{T} <: Manifold

The circle $\mathbb S^1$ as a manifold ere manifold represented by
real-valued data in $[-\pi,\pi)$ or complex-valued data $z\in \mathbb C$ of absolute value
$\lvert z\rvert = 1$.
# Constructor

    Circle(f=ℝ)

generates the Circle represented by angles, which alternatively can be set to use the field
`f=ℂ` to obtain the Circle as a manifold embedded in ℂ.
"""
struct Circle{F} <: Manifold where {F <: AbstractField} end
Circle(f::AbstractField=ℝ) = Circle{f}()


function check_manifold_point(M::Circle{ℝ},x; kwargs...)
    if !isapprox(sym_rem(x),x;kwargs...)
        return DomainError(x,"The point $(x) does not lie on $(M), since its is not in [-π,π).")
    end
    return nothing
end
function check_manifold_point(M::Circle{ℂ},x; kwargs...)
    if !isapprox(abs(x), 1.; kwargs...)
        return DomainError(abs(x), "The point $(x) does not lie on the $(M) since its norm is not 1.")
    end
    return nothing
end

function check_tangent_vector(M::Circle{ℝ},x,v; kwargs...)
    perr = check_manifold_point(M,x)
    return perr # if x is valid all v that are real numbers are valid
end
function check_tangent_vector(M::Circle{ℂ},x,v; kwargs...)
    perr = check_manifold_point(M,x)
    perr === nothing || return perr
    if !isapprox( abs(complex_dot(x,v)), 0.; kwargs...)
        return DomainError(abs(complex_dot(x,v)),
            "The value $(v) is not a tangent vector to $(x) on $(M), since it is not orthogonal in the embedding."
        )
    end
    return nothing
end

complex_dot(a::Number,b::Number) = (real(a)*real(b) + imag(a)*imag(b))
complex_dot(a,b) = abs(dot(a,b))

@doc doc"""
    distance(::Circle,x,y)

compute the distance on the [`Circle`](@ref) $\mathbb S^1$, which is
the absolute value of the symmetric remainder of `x` and `y` for the real-valued
case and the angle between both complex numbers in the Gaussian plane for the
complex-valued case.
"""
distance(::Circle{ℝ}, x::Real, y::Real) = abs(sym_rem(x-y))
distance(::Circle{ℝ}, x, y) = abs(sum(sym_rem.(x-y)))
distance(::Circle{ℂ}, x, y) = acos(clamp(complex_dot(x, y), -1, 1))

exp(::Circle{ℝ}, x::Real, v::Real) = sym_rem(x+v)
exp!(::Circle{ℝ}, y, x, v) = (y .= sym_rem(x+v))
function exp(M::Circle{ℂ}, x::Number, v::Number)
    θ = norm(M, x, v)
    return cos(θ)*x + usinc(θ)*v
end
function exp!(M::Circle{ℂ}, y, x, v)
    θ = norm(M, x, v)
    y .= cos(θ)*x + usinc(θ)*v
    return y
end

function flat!(::Circle, v::FVector{CotangentSpaceType}, x, w::FVector{TangentSpaceType})
    copyto!(v.data, w.data)
    return v
end
flat(M::Circle, x::Number, w::FVector{TangentSpaceType}) = FVector(CotangentSpace,w.data)

injectivity_radius(::Circle, args...) = π

@doc doc"""
    inner(M::Circle, x, w, v)

compute the inner product of the two tangent vectors `w,v` from the tangent plane at `x` on
the [`Circle`](@ref) $\mathbb S^1$ using the restriction of the metric from the embedding,
i.e.
````math
\langle v,w\rangle_x = w*v
````
for the real case and
````math
\langle v,w\rangle_x = v^\mathrm{T}w
````
for the complex case interpreting complex numbers in the Gaussian plane.
"""
@inline inner(::Circle{ℝ}, x, w, v) = dot(v,w)
@inline inner(::Circle{ℝ}, x::Real, w::Real, v::Real) = v*w
@inline inner(::Circle{ℂ}, x, w, v) = complex_dot(w, v)

inverse_retract(M::Circle, x, y) = inverse_retract(M, x, y, LogarithmicInverseRetraction())
inverse_retract(M::Circle, x, y, method::LogarithmicInverseRetraction) = log(M,x,y)

log(::Circle{ℝ}, x::Real, y::Real) = sym_rem(y-x)
log!(::Circle{ℝ}, v, x, y) = (v .= sym_rem(y-x))
function log(M::Circle{ℂ}, x::Number, y::Number)
    cosθ = complex_dot(x, y)
    if cosθ ≈ -1
        v = real(x) ≈ 1 ? 1im : 1+0im
        v = v - complex_dot(x, v)* x
        v *= π / norm(v)
    else
        cosθ = cosθ > 1 ? one(cosθ) : cosθ
        θ = acos(cosθ)
        v = (y - cosθ*x)/usinc(θ)
    end
    return project_tangent(M, x, v)
end
function log!(M::Circle{ℂ}, v, x, y)
    cosθ = complex_dot(x, y)
    if cosθ ≈ -1
        v .= real(x) ≈ 1 ? 1im : 1+0im
        v .= v - complex_dot(x, v)* x
        v *= π / norm(v)
    else
        cosθ = cosθ > 1 ? one(cosθ) : cosθ
        θ = acos(cosθ)
        v .-= (y - cosθ*x)/usinc(θ)
    end
    project_tangent!(M, v, x, v)
    return v
end

@doc doc"""
    manifold_dimension(M::Circle)

Return the dimension of the [`Circle`](@ref) $\mathbb S^1$, i.e. 1.
"""
manifold_dimension(::Circle) = 1

@doc doc"""
    mean(M::Circle, x::AbstractVector[, w::AbstractWeights])

Compute the Riemannian [`mean`](@ref mean(M::Manifold, args...)) of `x` on the
[`Circle`](@ref) $\mathbb S^1$ by the wrapped mean, i.e. the remainder of the
mean modulo 2π.
"""
mean(::Circle, args...)
mean(::Circle,x::Array{<:Real},w::AbstractVector; kwargs...) = sym_rem(sum(w.*x))

@inline norm(::Circle, x, v) = sum(abs.(v))

project_point(::Circle{ℝ}, x::Real) = sym_rem(x)
project_point(::Circle{ℂ}, x::Number) = x/abs(x)

project_point!(::Circle{ℝ}, x) = (x .= sym_rem(x))
project_point!(::Circle{ℂ}, x) = (x .= x/sum(abs.(x)))

project_tangent(::Circle{ℝ}, x::Real, v::Real) = v
project_tangent(::Circle{ℂ}, x::Number, v::Number) = v - complex_dot(x,v)*x

project_tangent!(::Circle{ℝ}, w, x, v) = (w .= v)
project_tangent!(::Circle{ℂ}, w, x, v) = (w .= v - complex_dot(x,v)*x)

retract(M::Circle,x,y) = retract(M, x, y, ExponentialRetraction())
retract(M::Circle,x,y,m::ExponentialRetraction) = exp(M,x,y)

representation_size(::Circle) = (1,)

function sharp!(M::Circle, v::FVector{TangentSpaceType}, x, w::FVector{CotangentSpaceType})
    copyto!(v.data, w.data)
    return v
end
sharp(M::Circle, x::Number, w::FVector{CotangentSpaceType}) = FVector(TangentSpace,w.data)

@doc doc"""
    sym_rem(x,[T=π])

symmetric remainder of `x` with respect to the interall 2*`T`, i.e.
`(x+T)%2T`, where the default for `T` is $\pi$
"""
sym_rem(x::N, T=π) where {N<:Number} = rem(x, convert(N,2*T),RoundNearest)
sym_rem(x, T=π) where N = rem.(x, Ref(convert(eltype(x),2*T)),Ref(RoundNearest))

vector_transport_to(::Circle{ℝ}, x::Real, v::Real, y::Real, ::ParallelTransport) = v
function vector_transport_to(M::Circle{ℂ}, x::Number, v::Number, y::Number, ::ParallelTransport)
    v_xy = log(M, x, y)
    vl = norm(M, x, v_xy)
    vto = v
    if vl > 0
        factor = 2*complex_dot(v, y)/(abs(x + y)^2)
        vto -= factor.*(x + y)
    end
    return vto
end
vector_transport_to!(::Circle{ℝ}, w, x, v, y, ::ParallelTransport) = (w .= v)
function vector_transport_to!(M::Circle{ℂ}, vto, x, v, y, ::ParallelTransport)
    v_xy = log(M, x, y)
    vl = norm(M, x, v_xy)
    vto .= v
    if vl > 0
        factor = 2*complex_dot(v, y)/(abs(x + y)^2)
        vto .-= factor.*(x + y)
    end
    return vto
end


vector_transport_along(M::Circle,x::Number,v::Number,c) = vector_transport_along!(M,zero(v),x,v,c)
function vector_transport_direction(M::Circle,x::Number,v::Number,vdir::Number,m::AbstractVectorTransportMethod)
    y = exp(M, x, vdir)
    return vector_transport_to(M, x, v, y, m)
end

zero_tangent_vector(::Circle,x::Number) = zero(x)
function zero_tangent_vector!(::Circle, v, x)
    fill!(v, 0)
    return v
end
