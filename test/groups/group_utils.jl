struct NotImplementedOperation <: AbstractGroupOperation end

struct NotImplementedManifold <: AbstractManifold{ℝ} end

struct NotImplementedGroupDecorator{M} <:
       AbstractDecoratorManifold{ℝ,TransparentGroupDecoratorType}
    manifold::M
end

struct DefaultTransparencyGroup{M,A<:AbstractGroupOperation} <:
       AbstractGroupManifold{ℝ,A,DefaultGroupDecoratorType}
    manifold::M
    op::A
end
