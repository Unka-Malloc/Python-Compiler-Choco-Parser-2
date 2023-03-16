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

    program = module.ops[0]

    defs = program.defs.ops
    stmts = program.stmts.ops

    for d in defs:
        if isinstance(d, FuncDef):
            f_name = d.func_name
            f_params = d.params.ops
            f_type = d.return_type.ops
            f_body = d.func_body.ops

            func_env[f_name] = False

            internal_typed_variables = {}

            for typed_var in f_params:
                internal_typed_variables[typed_var.var_name.data] = False

            for i in range(len(f_body)):
                if isinstance(f_body[i], Return):
                    for op in f_body[i].value.ops:
                        if isinstance(op, ExprName):
                            if op.id.data in internal_typed_variables.keys():
                                internal_typed_variables[op.id.data] = True

                    if i + 1 < len(f_body):
                        if f_body[i + 1:]:
                            print(f'[Warning] Dead code found: {UnreachableStatementsError()}')

                if isinstance(f_body[i], Literal):
                    if isinstance(f_body[i].value, IntegerAttr):
                        f_body[i].attributes["type_hint"] = choco_type.int_type

                        expr_env.append(f_body[i])

            if internal_typed_variables:
                for x in internal_typed_variables:
                    if not internal_typed_variables[x]:
                        print(f'[Warning] Dead code found:', str(UnusedArgumentError(x)).replace('"', ''))

        elif isinstance(d, VarDef):
            v_typed = d.typed_var.op
            v_value = d.literal.op

            vars_env[v_typed.var_name.data] = False

    for s in stmts:
        if isinstance(s, CallExpr):
            if s.func in func_env.keys():
                func_env[s.func] = True

            for arg in s.args.ops:
                if isinstance(arg, BinaryExpr):
                    lhs = arg.lhs
                    rhs = arg.rhs
                    operator = arg.op.data

                    if operator == "and":
                        for op in lhs.ops:
                            if isinstance(op.value, NoneAttr):
                                print(f'[Warning] Dead code found: {UnreachableExpressionError()}')
                            if isinstance(op.value, (BoolAttr, IntegerAttr, StringAttr)):
                                if op.value.data in (False, 0, ""):
                                    print(f'[Warning] Dead code found: {UnreachableExpressionError()}')

        if isinstance(s, Assign):
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
                if value.func in func_env.keys():
                    func_env[value.func] = True

    if expr_env:
        print(f'[Warning] Dead code found: The following expression is unused:')
        for e in expr_env:
            Printer().print_op(e)

    if func_env:
        for f_name in func_env.keys():
            if not func_env[f_name]:
                print(f'[Warning] Dead code found:', str(UnusedFunctionError(f_name)).replace('"', ''))

    if vars_env:
        for v_name in vars_env.keys():
            if not vars_env[v_name]:
                print(f'[Warning] Dead code found:', str(UnusedVariableError(v_name)).replace('"', ''))

    return module