from Cython.CodeWriter import CodeWriter as BaseWriter


class CodeWriter(BaseWriter):

    def visit_CFuncDefNode(self, node):
        """Handler for CFuncDefNode types"""
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
        """Handler for CFuncDeclaratorNode types"""
        # TODO: except, gil, etc.
        self.visit(node.base)
        self.put(u'(')
        self.comma_separated_list(node.args)
        self.endline(u'):')

    def visit_IndexNode(self, node):
        """Handler for IndexNode types"""
        self.visit(node.base)
        self.put("[")
        self.visit(node.index)
        self.put("]")

    def visit_NewExprNode(self, node):
        """Handler for NewExprNode types"""
        self.put("new ")
        self.visit(node.cppclass)

    def visit_DelStatNode(self, node):
        """Handler for DelExprNode types"""
        self.startline("del ")
        self.comma_separated_list(node.args)
        self.endline()

    def visit_BoolBinopNode(self, node):
        """Handler for BoolBinopNode"""
        self.visit(node.operand1)
        self.put(u" %s " % node.operator)
        self.visit(node.operand2)

    def visit_PrimaryCmpNode(self, node):
        """Handler for PrimaryCmpNode"""
        self.visit(node.operand1)
        self.put(u" %s " % node.operator)
        self.visit(node.operand2)

    def visit_SliceIndexNode(self, node):
        """Handler for SliceIndexNode"""
        self.visit(node.base)
        self.put("[")
        if node.start:
            self.visit(node.start)
        self.put(":")
        if node.stop:
            self.visit(node.stop)
        self.put("]")

    def visit_AmpersandNode(self, node):
        """Handler for AmpersandNode"""
        self.put("&")
        self.visit(node.operand)

    def visit_TypecastNode(self, node):
        """Handler for TypecastNode"""
        self.put("(<")
        self.visit(node.base_type)
        self.put(">")
        self.visit(node.operand)
        self.put(")")