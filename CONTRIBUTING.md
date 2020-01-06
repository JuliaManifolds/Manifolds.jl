# Contributing to Manifolds.jl

First off, thanks for taking the time to contribute. Any contribution is appreciated and welcome

The following is a set of guidelines to [Manifolds.jl](https://julianlsolvers.github.io/Manifolds.jl/).

#### Table of Contents

* [I just have a question](I-just-have-a-question)
* [How can I file an issue?](how-can-I-file-an-issue)
* [How Can I Contribute?](#how-can-I-contribute)

## I just have a question
The developers can most easily be reched either in the Julia Slack Channel [#manifolds](https://julialang.slack.com/archives/CP4QF0K5Z). You can apply for the Julia Slack [here](https://slackinvite.julialang.org) if you haven't joined yet. You can also ask your question on [discourse.julialang.org](https://discourse.julialang.org).

## How Can I file an Issue?
If you found a bug or want to propose a feature, we track our issues within the [GitHub repository](https://github.com/JuliaNLSolvers/Manifolds.jl/issues).

## How Can I Contribute ?

### Provide a new manifold
A main contribution you can provide is another manifold, that is not yet included in the
package. A manifold consists of  the parent type
[`Manifold`](https://julianlsolvers.github.io/Manifolds.jl/latest/interface.html#ManifoldsBase.Manifold)
from [`ManifoldsBase.jl`](https://julianlsolvers.github.io/Manifolds.jl/latest/interface.html).
This package also provides the main set of functions a manifold can/should implement.
Don't worry if you are only aware of a part of the functions. As long as the application
you have in mind only requires a subset of these functions, implement those. The
[`ManifoldsBase.jl`](https://julianlsolvers.github.io/Manifolds.jl/latest/interface.html)
interface provides concrete error messages for functions you might have missed.

One important detail is, that the interface often provides a mutating as well as a
non-mutating variant, see for example
[exp!](https://julianlsolvers.github.io/Manifolds.jl/latest/interface.html#ManifoldsBase.exp!-Tuple{Manifold,Any,Any,Any})
and
[exp](https://julianlsolvers.github.io/Manifolds.jl/latest/interface.html#Base.exp-Tuple{Manifold,Any,Any}).
The non-mutating one always falls back to use the mutating one, so in most cases it should
suffice to implement the mutating one, i.e.
[exp!](https://julianlsolvers.github.io/Manifolds.jl/latest/interface.html#ManifoldsBase.exp!-Tuple{Manifold,Any,Any,Any})
in this case.

Note that since the first argument is _always_ the [`Manifold`](https://julianlsolvers.github.io/Manifolds.jl/latest/interface.html#ManifoldsBase.Manifold), the mutated argument is always the second one in the signature.
In the example we have `exp(M, x, v)` for the exponential map and `exp!(M, y, v, x)` for the mutating one, that stores the result in `y`.

On the contrary the user will most likely look for the non-mutating version, when looking for help, so we recommend to add the documentation string to the non-mutating one, where all different signatures should be collected in one string. This can best be achieved by adding a docstring to the method with a general signature with the first argument being your manifold:
````julia
    struct MyManifold <: Manifold end
    @doc doc"""
        exp(M::MyManifold, x, v)
    
    Describe the function.
    """
    exp(::MyManifold, ::Any...)
````

### Code style

With the documentation we try to follow the
[documentation guidelines](https://docs.julialang.org/en/v1/manual/documentation/) from
the Julia documentation as well as follow the
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle).
It would be nice, if the `Manifold`s struct could contain a reference, where the general theory is from.
Any implemented function should be accompanied by its mathematical formulae, if a closed form exists.
Within the source code of one manifold, the type of the manifold should be the first element of the
file and an alphabetical order of the functions would be preferable.
