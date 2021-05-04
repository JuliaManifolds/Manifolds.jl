# Dime Tour, Rotations

This tutorial is meant to give the briefest of overviews on how to use Manifolds.jl in a manner familiar to those needing rigid body transforms.  This tutorial will introduce some of the common function calls needed to convert between the various data types and hopefully show the user something more about to combine some of the function calls listed elsewhere in the documentation.

## Rotations with SO(2)

Consider rotations on an xy-plane, commonly known (among others) as rotation matrices `R`, Direction Cosine Matrices `DCM`, `SpecialOrthogonal(2)` Lie Groups and associated Lie Algebra.  Let's load the necessary packages first:
```julia
using Manifolds
using LinearAlgebra
using StaticArrays
```

### Manifolds and Defaults

The associated manifolds and groups are defined by:
```julia
G = SpecialOrthogonal(2)
M = base_manifold(G)
@assert M == Rotations(2)
```

Pretty soon we will require some default definitions:
```julia
# default basis
e0 = DefaultOrthogonalBasis()
# default data type
p0 = @SMatrix [1.0 0; 0 1]

# Group identity element with zero aligned to the x-axis
xR0 = identity(G, p0)
```

Now let's say we want to define a manifold point `i` some rotation θ from the [`identity`](@ref) reference `xR0`.  This is easier to envision on `Rotations(2)`, while more complicated would likely use a generalized notion of distance between points instead.  For now considering a rotation, say
```julia
# + radians rotation from x-axis on plane to point i
xθi = π/6
```

### From Coordinates

To get our first Lie algebra element using the text book [`hat`](@ref), or equivaliently a more generalized [`get_vector`](@ref), function:
```julia
X_ = hat(G, xR0, xθi)             # specific definition
xXi = get_vector(G, xR0, xθi, e0)  # generalized definition
# 2×2 MMatrix{2, 2, Float64, 4} with indices SOneTo(2)×SOneTo(2):
#  0.0       -0.523599
#  0.523599   0.0
@assert isapprox( X_, xXi )
```

Note, in this case the same would work given the base manifold [`Rotations(2)`](@ref):
```julia
_X_ = hat(M, xR0, xθi)             # specific definition
_X = get_vector(M, xR0, xθi, e0)  # generalized definition
@assert _X_ == xXi; @assert _X == xXi
```

Now, let's place this algebra element on the manifold using the exponential map [`exp`](@ref):
```julia
xRi = exp(G, xR0, xXi)
# similarly for known underlying manifold
xRi_ = exp(M, xR0, xXi)

@assert isapprox( xRi, xRi_ )
```

### To Coordinates

The logarithm transform from the group back to algebra (or coordinates) is:
```julia
xXi_ = log(G, xR0, xRi)
xXi__ = log(M, xR0, xRi)
@assert xXi == xXi__
```

Similarly, the coordinate value can be extracted from the algebra using [`vee`](@ref), or directly from the group using the more generalized [`get_coordinates`](@ref):
```julia
# extracting coordinates using vee
xθi__ = vee(G, xR0, xXi_)[1]
_xθi__ = vee(M, xR0, xXi_)[1]

# OR, the preferred generalized get_coordinate function
xθi_ = get_coordinates(G, xR0, xXi_, e0)[1]
_xθi_ = get_coordinates(M, xR0, xXi_, e0)[1]

# confirm all versions are correct
@assert isapprox( xθi, xθi_ ); @assert isapprox( xθi, _xθi_ )
@assert isapprox( xθi, xθi__ ); @assert isapprox( xθi, _xθi__ )
```

### Actions and Operations

With the basics in hand on how to move between the coordinate, algebra, and group representations, let's briefly look at composition and application of points on the manifold.  For example, a `Rotation(n)` manifold is the mathematical representation, but points have an application purpose to retain information regarding a a specific rotation -- i.e. using Euler angles, or Euclidean(n) space to store rotation information quicly becomes problematic.  Other rotation represenation methods, including quaternions, Pauli matrices, etc., have similar features.

Therefore points from a manifold may well have an associated action which we [`apply`](@ref).  Consider rotating through `θ = π/6` three vectors `V` from their native domain `Euclidean(2)`, from the reference frame `a` to frame `b`.  Keeping with our two-dimensional example above:
```julia
aV1 = [1;0]
aV2 = [0;1]
aV3 = [10;10]

A_left = RotationAction(Euclidean(2), G)

bθa = π/6
bXa = get_vector(base_manifold(G), xR0, bθa, e0)

bRa = exp(G, R0, bXa)

for aV in [aV1; aV2; aV3]
  bV = apply(A_left, bRa, aV)
  # test we are getting the rotated vectors in Euclidean(2) as expected
  @assert isapprox( bV[1], norm(aV)*cos(bθa) )
  @assert isapprox( bV[2], norm(aV)*sin(bθa) )
end
```

!!! note
    In general, actions are usually non-commutative and the user must therefore be weary of [`LeftAction`](@ref) or [`RightAction`](@ref) needs.  As in this case, the default `LeftAction()` is used.

Finally, the actions (i.e. points from a manifold) can be [`compose`](@ref)d together.  Consider putting together two rotations `aRb` and `bRc` such that a single composite rotation `aRc` is found.  The next bit of code composes five rotations of `π/4` increments:
```julia
A_left = RotationAction(M, G)
aRi = deepcopy(xR0)

iθi_ = π/4
x_θ = get_vector(M, xR0, iθi_, e0) #hat(Rn, R0, θ)
iRi_ = exp(M, xR0, x_θ)

# do 5 times over:
# Ri_ = Ri*iRi_
for i in 1:5
  aRi = compose(A_left, aRi, iRi_)
end

# drop back to a algebra, then coordinate
aXi = log(G, xR0, aRi)
aθi = get_coordinates(G, xR0, aXi, e0)

# should wrap around to 3rd quadrant of xy-plane
@assert isapprox( -3π/4, aθi[1])
```

!!! warning
    `compose` or `apply` must be done with group (not algebra) elements.  This example shows how these two element types can easily be confused, since both the manifold group and algebra elements can have exactly the same data storage type -- i.e. a 2x2 matrix.
