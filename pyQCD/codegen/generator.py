from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import ast
import inspect
import re
import random
import string

class Generator(object):

    def __init__(self, tree):

        self.functions = {}
        self.local_types = {}
        self.member_types = {"lattice_": "Lattice*",
                             "evenIndices_": "vector<int>",
                             "oddIndices_": "vector<int>",
                             "operatorSize_": "int",
                             "boundaryConditions_":
                             "vector<vector<complex<double> > >"}
        self.args = {}
        self.locals = {}
        self.members = set(["lattice_", "evenIndices_", "oddIndices_",
                            "operatorSize_", "boundaryConditions_"])
        self.imports = {}

        self.collectfuncs(tree)

    def _handleobject(self, tree):

        try:
            return getattr(self, "_{}".format(tree.__class__.__name__))(tree)
        except AttributeError as e:
            print("Need to implement function _{}"
                  .format(tree.__class__.__name__))
            print("Error: {}".format(e))

            stack = inspect.stack()
            print("Location: {}, line {}".format(stack[1][3], stack[1][2]))
            print("{} members:".format(tree.__class__.__name__))
            for item in dir(tree):
                print(item)
            print()

    def funcgen(self, funcname):
        """Recurses through the function specified by funcname and assembles the
        function code in C++"""

        try:
            self.cur_loc_types = self.local_types[funcname]
        except KeyError:
            self.cur_loc_types = {}
        try:
            self.cur_args = self.args[funcname]
        except KeyError:
            self.cur_args = set()
        try:
            self.cur_locals = self.locals[funcname]
        except KeyError:
            self.cur_locals = set()
    
        code = self.translate_funcs(self._funcgen(self.functions[funcname]))
        declarations = self._declare_vars()

        self.local_types[funcname] = self.cur_loc_types
        self.args[funcname] = self.cur_args
        self.locals[funcname] = self.cur_locals

        return self.format("{}\n{}".format(declarations, code), 1)

    def _funcgen(self, tree):
        """Recurses through the supplied tree and assembles the function code
        in C++"""

        self.caller = self._funcgen
        
        if isinstance(tree, list):
            branch_codes = []
            for branch in tree:
                branch_codes.append(self._funcgen(branch))

            return "\n".join(branch_codes)

        else:
            return self._handleobject(tree)

    def _declare_vars(self):

        out = ""

        for var in self.cur_locals:
            try:
                array_search = re.findall("([\w\ ]+)\[([\w\d]+)\]",
                                          self.cur_loc_types[var])
                if len(array_search) > 0:
                    prefix = array_search[0][0]
                    suffix = "[{}]".format(array_search[0][1])
                else:
                    prefix = self.cur_loc_types[var]
                    suffix = ""
                out += "{} {}{};\n".format(prefix, var, suffix)
            except KeyError:
                raise KeyError("Variable {} does not have a specified type"
                               .format(var))

        return out

    def collectfuncs(self, tree):
        """Recurses through the supplied tree and appends function definitions
        to list of functions to translate"""

        self.caller = self.collectfuncs
            
        if isinstance(tree, list):
            for branch in tree:
                if isinstance(branch, ast.FunctionDef):
                    self.functions.update({branch.name: branch.body})
                    self.args[branch.name] = set()
                    for arg in branch.args.args:
                        if arg.id != "self":
                            self.args[branch.name].add(arg.id)
                    for dec in branch.decorator_list:
                        self._handledecorator(dec, True, branch.name)
                elif isinstance(branch, ast.ClassDef):
                    for dec in branch.decorator_list:
                        self._handledecorator(dec, False)
                    self.collectfuncs(branch)
                    self.classname = branch.name
                else:
                    self.collectfuncs(branch)
        else:
            return self._handleobject(tree)

    def _handledecorator(self, tree, func, decorated_func=None):
        name = self.caller(tree.func)

        if name == "types" and self.caller.__name__ == "collectfuncs":

            typedict = (getattr(self, "local_types")
                        if func
                        else getattr(self, "member_types"))

            if func:
                try:
                    typedict = typedict[decorated_func]
                except KeyError:
                    typedict = {}

            for kw in tree.keywords:
                value = self._Str(kw.value)
                try:
                    typedict.update({kw.arg: value})
                except KeyError:
                    typedict = {kw.arg: value}

            if func:
                getattr(self, "local_types")[decorated_func] = typedict
            else:
                setattr(self, "member_types", typedict)

    def _Module(self, tree):
        self.caller(tree.body)

    def _ClassDef(self, tree):
        self.caller(tree.body)

    def _FunctionDef(self, tree):
        return self.caller(tree.body)

    def _Name(self, tree):
        return tree.id

    def _Call(self, tree):
        func = self.caller(tree.func)
        args = [str(self.caller(arg)) for arg in tree.args]

        return "{}({})".format(func, ", ".join(args))

    def _Str(self, tree):
        return tree.s

    def _Assign(self, tree):

        if len(tree.targets) > 1:
            raise NotImplementedError("C++ does not support multiple assignment"
                                      "targets")
        
        target = self.caller(tree.targets)

        if "." in target:
            varname = target.split(".")[0]
        elif "->" in target:
            varname = target.split("->")[0]
        else:
            varname = target

        if (not varname in self.cur_args
            and not varname in self.cur_locals
            and varname != "this"):

            array_strip = re.findall("(\w+)\[[\w\d]+\]", varname)
            if len(array_strip) > 0:
                varname = array_strip[0]

            matrix_strip = re.findall("(\w+)\([\d\w,\ ]+\)", varname)
            if len(matrix_strip) > 0:
                varname = matrix_strip[0]
            
            self.cur_locals.add(varname)

        value = self.caller(tree.value)

        out = ""

        if type(tree.value) == ast.List or type(tree.value) == ast.ListComp:
            search_vector = re.findall("vector<(\w+)>",
                                       self.cur_loc_types[varname])
            num_elems = value.count(",") + 1
            temp_varname \
              = "temp{}".format("".join(random.sample(string.letters, 5)))
            if len(search_vector) > 0:
                base_type = search_vector[0]
                out += "{} {}[{}]".format(base_type, temp_varname, num_elems)
                out += " = {};\n".format(value)
                out += "{}.assign({}, {} + {});".format(target, temp_varname,
                                                        temp_varname, num_elems)

            search_array = re.findall("(\w+)\[(\d+)\]",
                                      self.cur_loc_types[varname])
            search_array \
              += [(r,) for r in re.findall("(\w+)\*",
                                           self.cur_loc_types[varname])]
            if len(search_array) > 0:
                base_type = search_array[0][0]
                out += "{} {}[{}]".format(base_type, temp_varname, num_elems)
                out += " = {};\n".format(value)
                out += "copy({}, {} + {}, {})".format(temp_varname,
                                                      temp_varname,
                                                      num_elems, target)

            return out
        
        else:
            return "{} = {};".format(target, value)

    def _Attribute(self, tree):

        attr = tree.attr
        value = self.caller(tree.value)

        if (type(tree.value) != ast.Attribute and value != "self"
            and not value in self.imports.values()):
            self.cur_locals.add(value)

        connector = "."
        if value == "self":
            value = "this"
            connector = "->"
        if "." in value:
            varname = value.split(".")[-1]
        elif "->" in value:
            varname = value.split("->")[-1]
        else:
            varname = value

        try:            
            if self.cur_loc_types[varname][-1] == "*":
                connector = "->"
        except KeyError:
            pass
        try:            
            if self.member_types[varname][-1] == "*":
                connector = "->"
        except KeyError:
            pass

        return "{}{}{}".format(value, connector, attr)

    def _Num(self, tree):
        return tree.n

    def _Return(self, tree):
        return "return {};".format(self.caller(tree.value))

    def _BinOp(self, tree):
        L = self.caller(tree.left)
        R = self.caller(tree.right)
        O = self.caller(tree.op)
        return "({} {} {})".format(L, O, R)

    def _Mult(self, tree):
        return "*"

    def _Add(self, tree):
        return "+"

    def _Div(self, tree):
        return "/"

    def _Sub(self, tree):
        return "-"

    def _Pow(self, tree):
        return "**"

    def _List(self, tree):

        list_values = []

        for elem in tree.elts:
            list_values.append(str(self.caller(elem)))

        return "{{{}}}".format(", ".join(list_values))

    def _Tuple(self, tree):

        list_values = []

        for elem in tree.elts:
            list_values.append(str(self.caller(elem)))

        return "{{{}}}".format(", ".join(list_values))

    def _ListComp(self, tree):

        out = "[{}".format(self.caller(tree.elt))

        for gen in tree.generators:
            out += " for {} in {}".format(self.caller(gen.target),
                                          self.caller(gen.iter))
            for if_clause in gen.ifs:
                out += " if {}".format((if_clause))

        out += "]"

        return "{{{}}}".format(str(eval(out))[1:-1])

    def _Compare(self, tree):
        return ""

    def _Expr(self, tree):
        return "{};".format(self.caller(tree.value))

    def _Subscript(self, tree):

        try:
            val_type = self.cur_loc_types[self.caller(tree.value)]
    
            eigen_types = ["Matrix3cd", "Matrix4cd", "VectorXcd"]
            bracket_template = "{}({})" if val_type in eigen_types else "{}[{}]"

        except KeyError:
            bracket_template = "{}[{}]"

        out = bracket_template.format(self.caller(tree.value),
                                      self.caller(tree.slice))
        
        return out.replace("[{", "(").replace("}]", ")")

    def _Index(self, tree):
        return self.caller(tree.value)

    def _For(self, tree):
        body =  self.caller(tree.body)
        iterator = self.caller(tree.iter)
        orelse = self.caller(tree.orelse)
        target = self.caller(tree.target)

        argsearch = []
        if iterator[:5] != "range":
            raise NotImplementedError("For loop iterator {} not supported"
                                      .format(iterator))

        args = [int(x) for x in re.findall("(\d+)", iterator)]
        if len(args) == 1:
            start, stop, incr = 0, args[0], 1
        elif len(args) == 2:
            start, stop, incr = args[0], args[1], 1
        elif len(args) == 3:
            start, stop, incr = tuple(args)
        else:
            raise NotImplementedError("Bad call to function range")

        comp = "<" if start < stop else ">"
        op = "+=" if start < stop else "-="

        header = ("for (int {} = {}; {} {} {}; {} {} {}) {{\n"
                  .format(target, start, target, comp, stop, target, op, incr))

        return "{}{}\n}}".format(header, body)

    def _Import(self, tree):
        for name in tree.names:
            self.imports[name.name] = name.asname or name.name

    def _AugAssign(self, tree):
        L = self.caller(tree.target)
        R = self.caller(tree.value)
        O = self.caller(tree.op)
        return "{} {}= {};".format(L, O, R)
                
    def translate_funcs(self, code):

        code = code.replace(".append", ".push_back")
        code = code.replace("{}.id4".format(self.imports['pyQCD']),
                            "Matrix4cd::Identity()")
        code = code.replace("{}.identity(4)".format(self.imports['numpy']),
                            "Matrix4cd::Identity()")
        code = code.replace("{}.identity(3)".format(self.imports['numpy']),
                            "Matrix3cd::Identity()")
        code = code.replace("{}.".format(self.imports['pyQCD']), "pyQCD::")

        hopping_search = re.findall("HoppingTerm\((\d+)\)", code)
        for res in hopping_search:
            old = "HoppingTerm({})".format(res)
            new = ("new HoppingTerm(boundaryConditions, lattice, {})"
                   .format(res))

            code = code.replace(old, new)

        hopping_search = re.findall("HoppingTerm\((\d+),\ *([\w\ ]+)\)", code)
        for res in hopping_search:
            old = "HoppingTerm({}, {})".format(*res)
            new = ("new HoppingTerm(boundaryConditions, lattice,"
                   " {}, {})".format(*res))

            code = code.replace(old, new)

        code = code.replace("{}.zeros({{3, 3}})".format(self.imports["numpy"]),
                            "Matrix3cd::Zero()")
        code = code.replace("{}.zeros({{3, 3}}, dtype=np.complex)"
                            .format(self.imports["numpy"]),
                            "Matrix3cd::Zero()")
            
        return code

    def format(self, code, start_indent=0):

        old_lines = code.split("\n")
        new_lines = []
        indent_count = start_indent
        
        for line in old_lines:
            try:
                if line[-1] == "{":
                    new_lines.append(indent_count * "  " + line)
                    indent_count += 1
                elif line[-1] == "}":
                    indent_count -= 1
                    new_lines.append(indent_count * "  " + line)
                else:
                    raise IndexError
            except IndexError:
                new_lines.append(indent_count * "  " + line)

        return "\n".join(new_lines)
