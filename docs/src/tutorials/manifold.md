# [How to implement your own manifold](@id manifold-tutorial)

```@meta
CurrentModule = ManifoldsBase
DocTestSetup  = quote
    using Manifolds
end
```

This tutorial demonstrates how to easily set your own manifold up within `Manifolds.jl`.

## Introduction

If you looked around a little and saw the [interface](../interface.md), the amount of functions and possibilities, it might seem that a manifold might take some time to implement.
This tutorial demonstrates that you can get your first own manifold quite fast and you only have to implement the functions you actually need. For this tutorial it would be helpful if you take a look at our [notation](../notation.md).
This tutorial assumes that you heard of the exponential map, tangent vectors and the dimension of a manifold. If not, please read for example [[do Carmo, 1992](#doCarmo1992)],
Chapter 3, first.

We start with two technical preliminaries. If you want to start directly, you can [skip](@ref manifold-tutorial-task) this paragraph and revisit it for two of the implementation details.

After that, we will

* [model](@ref manifold-tutorial-task) the manifold
* [implement](@ref manifold-tutorial-checks) two tests, so that points and tangent vectors can be checked for validity, for example also within [`ValidationManifold`](@ref),
* [implement](@ref manifold-tutorial-fn) two functions, the exponential map and the manifold dimension.

## [Technical preliminaries](@id manifold-tutorial-prel)

There are only two small technical things we need to explain at this point.
First of all our [`Manifold`](@ref)`{𝔽}` has a parameter `𝔽`.
This parameter indicates the [`number_system`](@ref) the manifold is based on, for example `ℝ` for real manifolds. It is important primarily for defining bases of tangent spaces.
See [`Symmetric`](@ref) as an example of defining both a real-valued and a complex-valued symmetric manifolds using one type.

Second, a main design decision of `Manifold.jl` is that most functions are implemented as mutating functions, i.e. as in-place-computations. There usually exists a non-mutating version that falls back to allocating memory and calling the mutating one. This means you only have to implement the mutating version, _unless_ there is a good reason to provide a special case for the non-mutating one, i.e. because in that case you know a far better performing implementation.

Let's look at an example. The exponential map $\exp_p\colon T_p\mathcal M \to \mathcal M$ that maps a tangent vector $X\in T_p\mathcal M$ from the tangent space at $p\in \mathcal M$ to the manifold.
The function [`exp`](@ref exp(M::Manifold, p, X)) has to know the manifold `M`, the point `p` and the tangent vector `X` as input, so to compute the resulting point `q` you need to call

```julia
q = exp(M, p, X)
```

If you already have allocated memory for the variable that should store the result, it is better to perform the computations directly in that memory and avoid reallocations. For example

```julia
q = similar(p)
exp!(M, q, p, X)
```

calls [`exp!`](@ref exp!(M::Manifold, q, p, X)), which modifies its input `q` and returns the resulting point in there.
Actually these two lines are (almost) the default implementation for [`exp`](@ref exp(M::Manifold, p, X)). [`allocate_result`](@ref) that is actually used there just calls `similar` for simple `Array`s.
Note that for a unified interface, the manifold `M` is _always_ the first parameter, and the variable the result will be stored to in the mutating variants is _always_ the second parameter.

Long story short: if possible, implement the mutating version [`exp!`](@ref exp!(M::Manifold, q, p, X)), you get the [`exp`](@ref exp(M::Manifold, p, X)) for free.
Many functions that build upon basic functions employ the mutating variant, too, to avoid reallocations.

## [Startup](@id manifold-tutorial-startup)

As a start, let's load `ManifoldsBase.jl` and import the functions we consider throughout this tutorial.
For implementing a manifold, loading the interface should suffice for quite some time.

```@example manifold-tutorial
using ManifoldsBase, LinearAlgebra, Test
import ManifoldsBase: check_manifold_point, check_tangent_vector, manifold_dimension, exp!
```

## [Goal](@id manifold-tutorial-task)

As an example, let's implement the sphere, but with a radius $r$.
Since this radius is a property inherent to the manifold, it will become a field of the manifold.
The second information, we want to store is the dimension of the sphere, for example whether it's the 1-sphere, i.e. the circle, represented by vectors $p\in\mathbb R^2$ or the 2-sphere in $\mathbb R^3$.
Since the latter might be something we want to [dispatch](https://en.wikipedia.org/wiki/Multiple_dispatch) on, we model it as a parameter of the type.

In general the `struct` of a manifold should provide information about the manifold, which are inherent to the manifold or has to be available without a specific point or tangent vector present.
This is -- most prominently -- a way to determine the manifold dimension.

For our example we define

```@example manifold-tutorial
"""
    MySphere{N} <: Manifold{ℝ}

Define an `n`-sphere of radius `r`. Construct by `MySphere(radius,n)`
"""
struct MySphere{N} <: Manifold{ManifoldsBase.ℝ} where {N}
    radius::Float64
end
MySphere(radius, n) = MySphere{n}(radius)
Base.show(io::IO, M::MySphere{n}) where {n} = print(io, "MySphere($(M.radius),$n)")
nothing #hide
```

Here, the last line just provides a nicer print of a variable of that type
Now we can already initialize our manifold that we will use later, the $2$-sphere of radius $1.5$.

```@example manifold-tutorial
S = MySphere(1.5, 2)
```

## [Checking points and tangents](@id manifold-tutorial-checks)

If we have now a point, represented as an array, we would first like to check, that it is a valid point on the manifold.
For this one can use the easy interface [`is_manifold_point`](@ref is_manifold_point(M::Manifold, p; kwargs...)). This internally uses [`check_manifold_point`](@ref check_manifold_point(M, p; kwargs...)).
This is what we want to implement.
We have to return the error if `p` is not on `M` and `nothing` otherwise.

We have to check two things: that a point `p` is a vector with `N+1` entries and it's norm is the desired radius.
To spare a few lines, we can use [short-circuit evaluation](https://docs.julialang.org/en/v1/manual/control-flow/#Short-Circuit-Evaluation-1) instead of `if` statements.
If something has to only hold up to precision, we can pass that down, too using the `kwargs...`.

```@example manifold-tutorial
function check_manifold_point(M::MySphere{N}, p; kwargs...) where {N}
    (size(p)) == (N+1,) || return DomainError(size(p),"The size of $p is not $((N+1,)).")
    if !isapprox(norm(p), M.radius; kwargs...)
        return DomainError(norm(p), "The norm of $p is not $(M.radius).")
    end
    return nothing
end
nothing #hide
```

Similarly, we can verify, whether a tangent vector `X` is valid. It has to fulfill the same size requirements and it has to be orthogonal to `p`. We can again use the `kwargs`, but also provide a way to check `p`, too.

```@example manifold-tutorial
function check_tangent_vector(M::MySphere, p, X, check_base_point = true, kwargs...)
    if check_base_point
        mpe = check_manifold_point(M, p; kwargs...)
        mpe === nothing || return mpe
    end
    size(X) != size(p) && return DomainError(size(X), "The size of $X is not $(size(p)).")
    if !isapprox(dot(p,X), 0.0; kwargs...)
        return DomainError(dot(p,X), "The tangent $X is not orthogonal to $p.")
    end
    return nothing
end
nothing #hide
```

to test points we can now use

```@example manifold-tutorial
is_manifold_point(S, [1.0,0.0,0.0]) # norm 1, so not on S, returns false
@test_throws DomainError is_manifold_point(S, [1.5,0.0], true) # only on R^2, throws an error.
p = [1.5,0.0,0.0]
X = [0.0,1.0,0.0]
# The following two tests return true
[ is_manifold_point(S, p); is_tangent_vector(S,p,X) ]
```

## [Functions on the manifold](@id manifold-tutorial-fn)

For the [`manifold_dimension`](@ref manifold_dimension(M::Manifold)) we have to just return the `N` parameter

```@example manifold-tutorial
manifold_dimension(::MySphere{N}) where {N} = N
manifold_dimension(S)
```

Note that we can even omit the variable name in the first line since we do not have to access any field or use the variable otherwise.

To implement the exponential map, we have to implement the formula for great arcs, given a start point `p` and a direction `X` on the $n$-sphere of radius $r$ the formula reads

````math
\exp_p X = \cos(\frac{1}{r}\lVert X \rVert)p + \sin(\frac{1}{r}\lVert X \rVert)\frac{r}{\lVert X \rVert}X.
````

Note that with this choice we for example implicitly assume a certain metric. This is completely fine. We only have to think about specifying a metric explicitly, when we have (at least) two different metrics on the same manifold.

An implementation of the mutation version, see the [technical note](@ref manifold-tutorial-prel), reads

```@example manifold-tutorial
function exp!(M::MySphere{N}, q, p, X) where {N}
    nX = norm(X)
    if nX == 0
        q .= p
    else
        q .= cos(nX/M.radius)*p + M.radius*sin(nX/M.radius) .* (X./nX)
    end
    return q
end
nothing #hide
```

A first easy check can be done taking `p` from above and any vector `X` of length `1.5π` from its tangent space. The resulting point is opposite of `p`, i.e. `-p`

```@example manifold-tutorial
q = exp(S,p, [0.0,1.5π,0.0])
[isapprox(p,-q); is_manifold_point(S,q)]
```

## [Conclusion](@id manifold-tutorial-outlook)

You can now just continue implementing further functions from the [interface](../interface.md),
but with just [`exp!`](@ref exp!(M::Manifold, q, p, X)) you for example already have

* [`geodesic`](@ref geodesic(M::Manifold, p, X)) the (not necessarily shortest) geodesic emanating from `p` in direction `X`.
* the [`ExponentialRetraction`](@ref), that the [`retract`](@ref retract(M::Manifold, p, X)) function uses by default.

For the [`shortest_geodesic`](@ref shortest_geodesic(M::Manifold, p, q)) the implementation of a logarithm [`log`](@ref ManifoldsBase.log(M::Manifold, p, q)), again better a [`log!`](@ref log!(M::Manifold, X, p, q)) is necessary.

Sometimes a default implementation is provided; for example if you implemented [`inner`](@ref inner(M::Manifold, p, X, Y)), the [`norm`](@ref norm(M, p, X)) is defined. You should overwrite it, if you can provide a more efficient version. For a start the default should suffice.
With [`log!`](@ref log!(M::Manifold, X, p, q)) and [`inner`](@ref inner(M::Manifold, p, X, Y)) you get the [`distance`](@ref distance(M::Manifold, p, q)), and so.

In summary with just these few functions you can already explore the first things on your own manifold. Whenever a function from `Manifolds.jl` requires another function to be specifically implemented, you get a reasonable error message.

## Literature

```@raw html
<ul>
<li id="doCarmo1992">
    [<a>doCarmo, 1992</a>]
    M. P. do Carmo,
    <emph>Riemannian Geometry</emph>,
    Birkhäuser Boston, 1992,
    ISBN: 0-8176-3490-8.
</li>
</ul>
```
