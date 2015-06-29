"""This module contains convenience functions for creating the various Cython
node types"""

from __future__ import absolute_import

from Cython.Compiler import ExprNodes, Nodes


def generate_type_node(typedef, funcs):
    """Iterate through builder functions and build a Cython Node instance."""

    stats = []
    for func in funcs:
        stats.append(func(typedef))

    return Nodes.CClassDefNode(None, class_name=typedef.name,
                               base_class_name=None, module_name=None,
                               body=Nodes.StatListNode(None, stats=stats))


class Builder(object):
    """Handles Cython context and provides functions to build Cython nodes."""

    def __init__(self, wrap_ptr=True):
        """Constructor for Builder. See help(Builder)."""
        self.wrap_ptr = True
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
        ret = ExprNodes.AttributeNode(None, attribute="instance",
                                      obj=ExprNodes.NameNode(None, name="self"))
        if self.wrap_ptr:
            ret = ExprNodes.IndexNode(
                None, index=ExprNodes.IntNode(None, value='0'), base=ret)
        declarator = Nodes.CFuncDeclaratorNode(
            None, args=[Nodes.CArgDeclNode(
                None,
                base_type=Nodes.CSimpleBaseTypeNode(None, name="self"),
                declarator=Nodes.CNameDeclaratorNode(None, name=""),
                default=None
            )],
            base=Nodes.CNameDeclaratorNode(None, name="cppobj")
        )
        return Nodes.CFuncDefNode(
            None, overridable=False, visibility="private", api=0,
            declarator=declarator, body=Nodes.ReturnStatNode(None, value=ret))


class ContainerBuilder(Builder):
    """Builder subclass for Container types."""

    def build_getitem(self, typedef):
        pass
