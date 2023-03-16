from typing import Callable

from choco.dialects.choco_ast import *
import re


def camel_to_snake(name):
    pattern = re.compile(r'(?<!^)(?=[A-Z])')
    return pattern.sub('_', name).lower()


def get_method(instance: object, method: str) -> Optional[Callable]:
    if not hasattr(instance, method):
        return None
    else:
        f = getattr(instance, method)
        if callable(f):
            return f
        else:
            return None


class Visitor:

    def visit_module_op(self, op):
        program = op.ops[0]

        self.visit_program(program)

    def visit_program(self, op):
        defs = op.defs.ops
        stmts = op.stmts.ops

        self.visit_defs(defs)
        self.visit_stmts(stmts)

    def visit_defs(self, op: Operation):
        pass

    def visit_stmts(self, op: Operation):
        pass

    def traverse(self, operation: Operation):
        class_name = camel_to_snake(type(operation).__name__)

        traverse = get_method(self, f"traverse_{class_name}")
        if traverse:
            traverse(operation)
        else:
            for r in operation.regions:
                for b in r.blocks:
                    for op in b.ops:
                        self.traverse(op)

        visit = get_method(self, f"visit_{class_name}")
        if visit:
            visit(operation)
