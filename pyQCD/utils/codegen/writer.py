from Cython.CodeWriter import CodeWriter as BaseWriter


class CodeWriter(BaseWriter):

    def visit_CFuncDefNode(self, node):
        if 'inline' in node.modifiers:
            return
        if node.overridable:
            self.startline(u'cpdef ')
        else:
            self.startline(u'cdef ')
        if node.visibility != 'private':
            self.put(node.visibility)
            self.put(u' ')
        if node.api:
            self.put(u'api ')
        self.visit(node.declarator)
        self.indent()
        self.visit(node.body)
        self.dedent()

    def visit_CFuncDeclaratorNode(self, node):
        # TODO: except, gil, etc.
        self.visit(node.base)
        self.put(u'(')
        self.comma_separated_list(node.args)
        self.endline(u'):')
