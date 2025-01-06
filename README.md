# LearningJAX

`eqx1_Linear.ipynb`:
- eqx.Linear with autograd
- writing a custom linear layer with jax (abc.ABC, abc.abstractmethod, @dataclasses.dataclass) (all eqx.modules are written in this way)
- `py1_decorators.ipynb`: essentially a decorator is a wrapper function which adds (modifies) functionality to the original function
- `py2_dataclasses.ipynb`:
  - abstract classes are meant to be inherited by other classes
  - abstract methods are meant to enforce the overwritting by the inheriting classes 
  - the dataclass decorator is a shortcut to create classes with __init__, __repr__, __eq__, __hash__ methods

`jax0_types.ipynb`: 
- jax.Array and jnp.ndarray are the same
- an array with one element has a shape of (1,) while the value of the element is a scalar and has a shape of ()

`jax1_pytrees.ipynb`:
- pytrees: lists, tuples, dicts, namedtuple, None, OrderedDict, dataclasses
- can create custom pytrees with tree_flatten and tree_unflatten
- instance.tree_flatten() and jax.tree_util.tree_flatten(instance) are essentially the same while having one difference: 
    - instance.tree_flatten() returns a tuple of trees and the auxiliary data
    - jax.tree_util.tree_flatten(instance) returns a list of trees and the auxiliary data accompanied by the treedef

`jax2_jit.ipynb`: basic usage of jax.jit --> <font color='red'> TODO </font>

`jax3_jit_classes.ipynb`:
- you cannot jit a class method directly, the tracer will consider the class attributes as constants
- workaround: use a wrapper function to which you pass the class parameters 
    - the wrapper will unflatten the instance
    - you can then return the modified instance parameters and modify the instance outside the wrapper 