from io import StringIO

from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Block
from xdsl.printer import Printer

from choco.ast_visitor import Visitor
from choco.dialects.choco_ast import *


class DeadCodeError(Exception):
    pass


@dataclass
class UnreachableStatementsError(DeadCodeError):
    """Raised when some statements are unreachable."""

    def __str__(self) -> str:
        return "Program contains unreachable statements."


@dataclass
class UnreachableExpressionError(DeadCodeError):
    """Raised when parts of an expression is unreachable."""

    def __str__(self) -> str:
        return "Program contains unreachable expressions."


@dataclass
class UnusedStoreError(DeadCodeError):
    """Raised when a store operation is unused."""
    op: Assign

    def __str__(self) -> str:
        stream = StringIO()
        printer = Printer(stream=stream)
        print("The following store operation is unused: ", file=stream)
        printer.print_op(self.op)
        return stream.getvalue()


@dataclass
class UnusedVariableError(DeadCodeError):
    """Raised when a variable is unused."""

    name: str

    def __str__(self) -> str:
        return f"The following variable is unused: {self.name}."


@dataclass
class UnusedArgumentError(DeadCodeError):
    """Raised when a function argument is unused."""
    name: str

    def __str__(self) -> str:
        return f"The following function argument is unused: {self.name}."


@dataclass
class UnusedFunctionError(DeadCodeError):
    """Raised when a function is unused."""
    name: str

    def __str__(self) -> str:
        return f"The following function is unused: {self.name}."


@dataclass
class UnusedExpressionError(DeadCodeError):
    """Raised when the result of an expression is unused."""

    expr: Operation

    def __str__(self) -> str:
        stream = StringIO()
        printer = Printer(stream=stream)
        print("The following expression is unused: ", file=stream)
        printer.print_op(self.expr)
        return stream.getvalue()


def warn_dead_code(_: MLContext, module: ModuleOp) -> ModuleOp:
    # TODO: check for dead code in `module`, and raise the corresponding exception
    # if some dead code was found.

    expr_env = []
    func_env = {}
    vars_env = {}
    vars_assign = {}

    def visit_literal(node):
        if isinstance(node.value, NoneAttr):
            node.attributes["type_hint"] = choco_type.none_type
            expr_env.append(node)

        if isinstance(node.value, BoolAttr):
            node.attributes["type_hint"] = choco_type.bool_type
            expr_env.append(node)

        if isinstance(node.value, IntegerAttr):
            node.attributes["type_hint"] = choco_type.int_type
            expr_env.append(node)

        if isinstance(node.value, StringAttr):
            node.attributes["type_hint"] = choco_type.str_type
            expr_env.append(node)

    def visit_var_def(node):
        v_typed = node.typed_var.op
        v_value = node.literal.op

        vars_env[v_typed.var_name.data] = False

    def visit_func_def(node):
        f_name = node.func_name
        f_params = node.params.ops
        f_type = node.return_type.ops
        f_body = node.func_body.ops

        func_env[f_name] = False

        func_domain_vars = {}

        for typed_var in f_params:
            func_domain_vars[typed_var.var_name.data] = False

        for i in range(len(f_body)):
            if isinstance(f_body[i], Return):
                for op in f_body[i].value.ops:
                    if isinstance(op, ExprName):
                        if op.id.data in func_domain_vars.keys():
                            func_domain_vars[op.id.data] = True

                if i + 1 < len(f_body):
                    if f_body[i + 1:]:
                        print(f'[Warning] Dead code found: {UnreachableStatementsError()}')

                break

            if isinstance(f_body[i], Literal):
                visit_literal(f_body[i])

        if func_domain_vars:
            for x in func_domain_vars:
                if not func_domain_vars[x]:
                    print(f'[Warning] Dead code found:', str(UnusedArgumentError(x)).replace('"', ''))

    def visit_assign_expr(node):
        target = s.target.op
        value = s.value.op

        if isinstance(target, ExprName):
            if target.id.data in vars_assign.keys():
                print(f'[Warning] Dead code found: The following store operation is unused:')
                Printer().print_op(vars_assign[target.id.data])
            else:
                vars_assign[target.id.data] = s

            if target.id.data in vars_env.keys():
                vars_env[target.id.data] = True

        if isinstance(value, CallExpr):
            visit_call_expr(value)

    def visit_call_expr(node):
        if node.func in func_env.keys():
            func_env[node.func] = True

        for arg in node.args.ops:
            if isinstance(arg, ExprName):
                if arg.id.data in vars_assign.keys():
                    del vars_assign[arg.id.data]

            if isinstance(arg, BinaryExpr):
                visit_binary_op(arg)

    def visit_binary_op(node):
        lhs = node.lhs
        rhs = node.rhs
        operator = node.op.data

        if operator == '==':
            if len(lhs.ops) == 1 and len(rhs.ops) == 1:
                if lhs.ops[0].value == rhs.ops[0].value:
                    return True
                else:
                    return False

        if operator == '!=':
            if len(lhs.ops) == 1 and len(rhs.ops) == 1:
                if lhs.ops[0].value != rhs.ops[0].value:
                    return True
                else:
                    return False

        if operator == '>':
            if len(lhs.ops) == 1 and len(rhs.ops) == 1:
                if lhs.ops[0].value > rhs.ops[0].value:
                    return True
                else:
                    return False

        if operator == '<':
            if len(lhs.ops) == 1 and len(rhs.ops) == 1:
                if lhs.ops[0].value < rhs.ops[0].value:
                    return True
                else:
                    return False

        if operator == '>=':
            if len(lhs.ops) == 1 and len(rhs.ops) == 1:
                if lhs.ops[0].value >= rhs.ops[0].value:
                    return True
                else:
                    return False

        if operator == '<=':
            if len(lhs.ops) == 1 and len(rhs.ops) == 1:
                if lhs.ops[0].value <= rhs.ops[0].value:
                    return True
                else:
                    return False

        if operator == "and":
            for op in lhs.ops:
                if isinstance(op, BinaryExpr):
                    if not visit_binary_op(op):
                        print(f'[Warning] Dead code found: {UnreachableExpressionError()}')

                if isinstance(op, Literal):
                    if not op.value.data:
                        print(f'[Warning] Dead code found: {UnreachableExpressionError()}')

        if operator == "or":
            for op in lhs.ops:
                if isinstance(op, BinaryExpr):
                    visit_binary_op(op.value)

                if isinstance(op, Literal):
                    if op.value.data:
                        print(f'[Warning] Dead code found: {UnreachableExpressionError()}')

    def visit_if_expr(node):
        cond = node.cond.op
        then = node.then.ops
        orelse = node.orelse.ops

        if isinstance(cond, NoneAttr):
            print(f'[Warning] Dead code found: {UnreachableExpressionError()}')

        if isinstance(cond, BoolAttr):
            if not cond.value.data:
                print(f'[Warning] Dead code found: {UnreachableExpressionError()}')

        if isinstance(cond, IntegerAttr):
            if not cond.value.data:
                print(f'[Warning] Dead code found: {UnreachableExpressionError()}')

        if isinstance(cond, StringAttr):
            if not cond.value.data:
                print(f'[Warning] Dead code found: {UnreachableExpressionError()}')

        if isinstance(cond, BinaryExpr):
            visit_binary_op(cond)

    program = module.ops[0]

    defs = program.defs.ops
    stmts = program.stmts.ops

    for d in defs:
        if isinstance(d, FuncDef):
            visit_func_def(d)

        if isinstance(d, VarDef):
            visit_var_def(d)

    for s in stmts:
        # Unused Expression
        if isinstance(s, Literal):
            visit_literal(s)

        if isinstance(s, BinaryExpr):
            expr_env.append(s)

        if isinstance(s, CallExpr):
            visit_call_expr(s)

        if isinstance(s, Assign):
            visit_assign_expr(s)

        if isinstance(s, If):
            if not visit_if_expr(s):
                print(f'[Warning] Dead code found: {UnreachableExpressionError()}')


    if expr_env:
        print(f'[Warning] Dead code found: The following expression is unused:')
        for e in expr_env:
            Printer().print_op(e)

    if func_env:
        for f in func_env.keys():
            if not func_env[f]:
                print(f'[Warning] Dead code found:', str(UnusedFunctionError(f)).replace('"', ''))

    if vars_env:
        for v in vars_env.keys():
            if not vars_env[v]:
                print(f'[Warning] Dead code found:', str(UnusedVariableError(v)).replace('"', ''))

    if vars_assign:
        print(f'[Warning] Dead code found: The following store operation is unused: ')
        for k in vars_assign.keys():
            Printer().print_op(vars_assign[k])

    return module