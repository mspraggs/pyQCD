"""This module contains convenience functions for creating the various Cython
node types"""

from __future__ import absolute_import

import io

from Cython.Compiler import ExprNodes, Nodes
from Cython.Compiler.Main import Context
from Cython.Compiler.Parsing import p_module
from Cython.Compiler.Scanning import PyrexScanner, StringSourceDescriptor
from Cython.Compiler.Symtab import ModuleScope


def generate_type_node(typedef, builder):
    """Iterate through builder functions and build a Cython Node instance."""

    stats = []
    for feature in builder.features:
        func = getattr(builder, "build_{}".format(feature))
        stats.append(func(typedef))

    return Nodes.CClassDefNode(None, class_name=typedef.name,
                               base_class_name=None, module_name=None,
                               body=Nodes.StatListNode(None, stats=stats))


class Builder(object):
    """Handles Cython context and provides functions to build Cython nodes."""

    def __init__(self, wrap_ptr=True):
        """Constructor for Builder. See help(Builder)."""
        self.wrap_ptr = wrap_ptr
        self.features = [attr[6:] for attr in dir(self)
                         if attr.startswith("build_")]

    def build_instance(self, typedef):
        """Create an instance attribute for cdef classes to wrap a C++ object"""
        name_declarator = Nodes.CNameDeclaratorNode(None, name="instance")
        if self.wrap_ptr:
            declarators = [Nodes.CPtrDeclaratorNode(None, base=name_declarator)]
        else:
            declarators = [name_declarator]
        return Nodes.CVarDefNode(
            None, declarators=declarators,
            base_type=Nodes.CSimpleBaseTypeNode(None, name=typedef.cname))

    def build_cppobj(self, typedef):
        """Create cppobj member function to return instance"""
        declarator = Nodes.CFuncDeclaratorNode(
            None, args=generate_simple_args("self"),
            base=Nodes.CNameDeclaratorNode(None, name="cppobj")
        )
        return Nodes.CFuncDefNode(
            None, overridable=False, visibility="private", api=0,
            declarator=declarator,
            body=Nodes.ReturnStatNode(
                None, value=typedef.instance_raw_accessor("self")))

    def build_cinit(self, typedef):
        """Create __cinit__ method"""
        type_node = Nodes.CSimpleBaseTypeNode(None, name=typedef.cname)
        func = (ExprNodes.NewExprNode(None, cppclass=type_node)
                if self.wrap_ptr else type_node)
        rhs_node = ExprNodes.SimpleCallNode(None, function=func, args=[])
        lhs_node = typedef.instance_raw_accessor("self")
        body = Nodes.SingleAssignmentNode(None, lhs=lhs_node, rhs=rhs_node)
        return Nodes.DefNode(None, body=body, name="__cinit__",
                             args=generate_simple_args("self"))

    def build_dealloc(self, typedef):
        """Create __dealloc__ method"""
        if self.wrap_ptr:
            body = Nodes.DelStatNode(None, args=[
                ExprNodes.AttributeNode(
                    None, attribute="instance",
                    obj=ExprNodes.NameNode(None, name="self"))
            ])
        else:
            body = Nodes.PassStatNode(None)
        return Nodes.DefNode(None, body=body, name="__dealloc__",
                             args=generate_simple_args("self"))


class ContainerBuilder(Builder):
    """Builder subclass for Container types."""

    def build_buffer_shape(self, typedef):
        """Create a buffer_shape attribute for use with buffer protocol"""
        return generate_simple_array_def("Py_ssize_t", "buffer_shape",
                                         str(typedef.buffer_ndims))

    def build_buffer_strides(self, typedef):
        """Create a buffer_shape attribute for use with buffer protocol"""
        return generate_simple_array_def("Py_ssize_t", "buffer_strides",
                                         str(typedef.buffer_ndims))

    def build_to_numpy(self, typedef):
        """Create member function to return numpy buffer accessor"""
        src = ("arr = np.asarray(self)\n"
               "arr.dtype = complex\n"
               "return arr")
        body = parse_string(src)
        return Nodes.DefNode(None, body=body, name="to_numpy",
                             args=generate_simple_args("self"))

    def build_getitem(self, typedef):
        """Create __getitem__ member function node"""
        # TODO: Need some index validation code in here somewhere
        clauses = []
        len_sum_expr = None
        int_len_sum = 0
        cobj = typedef.instance_val_accessor("self")
        parent_type = typedef
        parent_sum_expr = ExprNodes.IntNode(None, value='0')
        # Loop through num_dims and TypeDef descriptors
        # ndims is expression node giving the length of the tuple required
        # in order to return the type specified by TypeDef t.
        for ndims, elemtype in typedef.accessor_info:
            lengths = generate_lookup_lengths(int_len_sum, len_sum_expr, ndims)
            int_len_sum, len_sum_expr, total_sum_expr = lengths
            condition = parse_string("type(index) is tuple and "
                                     "len(index) is x").expr
            condition.operand2.operand2 = total_sum_expr
            if_body = parse_string("out = Foo()\n"
                                   "return out")
            if_body.stats[0].rhs.function.name = elemtype.name

            cobj = parent_type.ctype_elem_access(
                cobj, ExprNodes.NameNode(None, name="index"), parent_sum_expr)
            assignment = Nodes.SingleAssignmentNode(
                None, lhs=typedef.instance_val_accessor("out"), rhs=cobj)
            if_body.stats.insert(1, assignment)
            clauses.append(Nodes.IfClauseNode(None, condition=condition,
                                              body=if_body))
            parent_type = elemtype
            parent_sum_expr = total_sum_expr
        body = Nodes.IfStatNode(None, if_clauses=clauses, else_clause=None)
        return Nodes.DefNode(None, body=body, name="__getitem__",
                             args=generate_simple_args("self", "index"))

    def build_setitem(self, typedef):
        """Create __setitem__ member function node"""
        # TODO: Need some index validation code in here somewhere
        clauses = []
        len_sum_expr = None
        int_len_sum = 0
        cobj = typedef.instance_val_accessor("self")
        parent_type = typedef
        parent_sum_expr = ExprNodes.IntNode(None, value='0')
        # Loop through num_dims and TypeDef descriptors
        # ndims is expression node giving the length of the tuple required
        # in order to return the type specified by TypeDef t.
        for ndims, elemtype in typedef.accessor_info:
            lengths = generate_lookup_lengths(int_len_sum, len_sum_expr, ndims)
            int_len_sum, len_sum_expr, total_sum_expr = lengths
            condition = parse_string("type(index) is tuple and "
                                     "len(index) is x and "
                                     "type(value) is y").expr
            condition.operand2.operand1.operand2 = total_sum_expr
            condition.operand2.operand2.operand2 = ExprNodes.NameNode(
                None, name=elemtype.name)
            if_body = parse_string("cdef Foo* val\n"
                                   "val = &blah\n"
                                   "val[0] = blah")
            if_body.stats[0].base_type.name = elemtype.cname
            if_body.stats[1].rhs.operand = parent_type.ctype_elem_access(
                cobj, ExprNodes.NameNode(None, name="index"), parent_sum_expr
            )
            if_body.stats[2].rhs = elemtype.instance_val_accessor("value", True)

            cobj = parent_type.ctype_elem_access(
                cobj, ExprNodes.NameNode(None, name="index"), parent_sum_expr)
            clauses.append(Nodes.IfClauseNode(None, condition=condition,
                                              body=if_body))
            parent_type = elemtype
            parent_sum_expr = total_sum_expr
        body = Nodes.IfStatNode(None, if_clauses=clauses, else_clause=None)
        return Nodes.DefNode(None, body=body, name="__setitem__",
                             args=generate_simple_args("self", "index",
                                                       "value"))


def parse_string(src):
    """Parse a string into a Cython node tree, then return it"""

    desc = StringSourceDescriptor("", src)
    stream = io.StringIO(unicode(src))
    context = Context(["."], [])
    scope = ModuleScope("", None, context)
    scanner = PyrexScanner(stream, desc, source_encoding="UTF-8", scope=scope,
                           context=context)
    tree = p_module(scanner, 0, "").body
    tree.pos = None
    return tree


def generate_simple_array_def(typename, varname, ndims):
    """Generate a CVarDefNode using the specified parameters"""
    name_declarator = Nodes.CNameDeclaratorNode(None, name=varname)
    array_declarator = Nodes.CArrayDeclaratorNode(
        None, base=name_declarator,
        dimension=ExprNodes.IntNode(None, value=ndims))
    return Nodes.CVarDefNode(
        None, declarators=[array_declarator],
        base_type=Nodes.CSimpleBaseTypeNode(None, name=typename))


def generate_simple_args(*args):
    """Generate a list of CArgDeclNode objects for simple positional args"""
    out = []
    for arg in args:
        out.append(Nodes.CArgDeclNode(
            None, declarator=Nodes.CNameDeclaratorNode(None, name=arg),
            default=None, base_type=Nodes.CSimpleBaseTypeNode(None, name=None)))
    return out


def generate_lookup_lengths(int_len_sum, len_sum_expr, length):
    """Examine expression node length and return updated length expressions."""
    # Keep track of when length is an IntNode, as we can just add these
    # values up
    if isinstance(length, ExprNodes.IntNode):
        int_len_sum += int(length.value)
    else:
        # If l is a different kind of expression, add it to len_sum_expr
        if len_sum_expr is not None:
            len_sum_expr = ExprNodes.AddNode(None, operator='+',
                                             operand1=len_sum_expr,
                                             operand2=length)
        else:
            len_sum_expr = length
    int_len_sum_expr = ExprNodes.IntNode(None, value=str(int_len_sum))
    # Now we need to determine whether to add the two length expressions
    # together or not
    if int_len_sum and len_sum_expr:
        total_len_expr = ExprNodes.AddNode(
            None, operator='+', operand1=len_sum_expr,
            operand2=int_len_sum_expr)
    elif int_len_sum:
        total_len_expr = int_len_sum_expr
    else:
        total_len_expr = len_sum_expr
    return int_len_sum, len_sum_expr, total_len_expr
