from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os

import jinja2

from .generator import *

template_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            'templates')

function_dict = {"apply": "apply",
                 "apply_hermitian": "apply_herm",
                 "make_hermitian": "make_herm",
                 "apply_even_even_inv": "apply_even_even_inv",
                 "apply_odd_odd": "apply_odd_odd",
                 "apply_even_odd": "apply_even_odd",
                 "apply_odd_even": "apply_odd_even"}

def gen_from_src(input_file, header_file="linop.hpp",
                 source_file="linop.cpp"):
    with open(input_file) as f:
        tree = ast.parse(f.read())

    dest_stem = (os.path.split(input_file)[1][:-3]
                 if input_file.endswith(".py")
                 else os.path.split(input_file)[1])

    template_args = make_template_args(tree)
    template_args.update(include_guard=dest_stem.upper())
    template_args.update(include_name=dest_stem)
        
    header, source = generate(template_args, header_file, source_file)

    with open("{}.hpp".format(dest_stem), 'w') as f:
        f.write(header)
    with open("{}.cpp".format(dest_stem), 'w') as f:
        f.write(source)

def generate(template_args, header_file="linop.hpp",
             source_file="linop.cpp"):
    header_file = os.path.join(template_dir, header_file)
    source_file = os.path.join(template_dir, source_file)

    with open(header_file) as f:
        header_template = jinja2.Template(f.read())
    with open(source_file) as f:
        source_template = jinja2.Template(f.read())

    return (header_template.render(**template_args),
            source_template.render(**template_args))

def make_template_args(syntax_tree):
    generator = Generator(syntax_tree)
    template_args = {}
    template_args.update(ctor_body=generator.funcgen("__init__"))
    for funcname, keyword in function_dict.items():
        template_args.update({"{}_body".format(keyword):
                              generator.funcgen(funcname)})
    template_args.update(destructibles=generator.destructibles())

    if "HoppingTerm*" in generator.member_types.values():
        hopper = [key for key, value in generator.member_types.items()
                  if value == "HoppingTerm*"][0]
        even_odd_handling \
          = ("this->evenIndices_ = this->{}->getEvenIndices();\n"
             "this->oddIndices_ = this->{}->getOddIndices();"
             .format(hopper, hopper))
        even_odd_handling = generator.format(even_odd_handling, 1)
    else:
        even_odd_handling = ""
    
    template_args.update(ctor_args=generator.declare_args('__init__'))
    for funcname, keyword in function_dict.items():
        template_args.update({"{}_arg".format(keyword):
                              list(generator.args[funcname])[0]})
    template_args.update(member_vars=generator.declare_members())
    template_args.update(class_name=generator.classname)
    template_args.update(even_odd_handling=even_odd_handling)

    return template_args
