from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from functools import reduce
from typing import Dict, List, Optional, Tuple, Union

from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import MLContext, Operation, Attribute
from choco.ast_visitor import Visitor
from choco.dialects import choco_ast, choco_type
from choco.semantic_error import SemanticError


class Type(ABC):

    @staticmethod
    def from_op(op: Operation) -> 'Type':
        if isinstance(op, choco_ast.TypeName):
            name = op.type_name.data  #type: ignore
            if name == "int":
                return int_type
            if name == "bool":
                return bool_type
            if name == "str":
                return str_type
            if name == "<None>":
                return none_type
            if name == "<Empty>":
                return empty_type
            if name == "object":
                return object_type
            raise Exception(f"Found type with unknown name `{name}'")
        if isinstance(op, choco_ast.ListType):
            elem_type = Type.from_op(op.elem_type.op)
            return ListType(elem_type)
        raise Exception(f'Found {op}, expected TypeName or ListType')

    @staticmethod
    def from_attribute(attr: Attribute) -> 'Type':
        if isinstance(attr, choco_type.NamedType):
            name: str = attr.type_name.data  # type: ignore
            if name == "int":
                return int_type
            if name == "bool":
                return bool_type
            if name == "str":
                return str_type
            if name == "<None>":
                return none_type
            if name == "<Empty>":
                return empty_type
            if name == "object":
                return object_type
            raise Exception(f"Found type with unknown name `{name}'")
        if isinstance(attr, choco_type.ListType):
            elem_type = Type.from_attribute(attr.elem_type)  # type: ignore
            return ListType(elem_type)
        raise Exception(f'Found {attr}, expected TypeName or ListType')


@dataclass
class BasicType(Type):
    name: str


@dataclass
class ListType(Type):
    elem_type: Type


class BottomType(Type):  # "⊥"
    pass


class ObjectType(Type):  # "object"
    pass


@dataclass
class FunctionType(Type):  # input0 x ... x inputN -> output
    inputs: List[Type]
    output: Type


int_type = BasicType("int")
bool_type = BasicType("bool")
str_type = BasicType("str")
none_type = BasicType("<None>")
empty_type = BasicType("<Empty>")
bottom_type = BottomType()
object_type = ObjectType()


def to_attribute(t: Type) -> Attribute:
    if t == int_type:
        return choco_type.int_type
    elif t == bool_type:
        return choco_type.bool_type
    elif t == str_type:
        return choco_type.str_type
    elif t == none_type:
        return choco_type.none_type
    elif t == empty_type:
        return choco_type.empty_type
    elif t == object_type:
        return choco_type.object_type
    elif isinstance(t, ListType):
        elem_type = to_attribute(t.elem_type)
        return choco_type.ListType([elem_type])
    else:
        raise Exception(f"Can't translate {t} into an attribute")


def join(t1: Type, t2: Type) -> Type:
    if is_assignment_compatible(t1, t2):
        return t2
    if is_assignment_compatible(t2, t1):
        return t1
    else:
        return object_type


def is_subtype(t1: Type, t2: Type) -> bool:
    if t1 == t2:
        return True
    elif t2 == object_type and (t1 == int_type or t1 == bool_type
                                or t1 == str_type or isinstance(t1, ListType)):
        return True
    elif t1 == none_type and t2 == object_type:
        return True
    elif t1 == empty_type and t2 == object_type:
        return True
    elif t1 == bottom_type:
        return True
    else:
        return False


def is_assignment_compatible(t1: Type, t2: Type) -> bool:
    if is_subtype(t1, t2) and t1 != bottom_type:
        return True
    elif (
            t1 == none_type
    ) and t2 != int_type and t2 != bool_type and t2 != str_type and t2 != bottom_type:
        return True
    elif isinstance(t2, ListType) and t1 == empty_type:
        return True
    elif isinstance(t2, ListType) and isinstance(t1, ListType) and \
            t1.elem_type == none_type and \
            is_assignment_compatible(none_type, t2.elem_type):
        return True
    else:
        return False


def check_assignment_compatibility(t1: Type, t2: Type):
    if not is_assignment_compatible(t1, t2):
        raise SemanticError(
            f" Expected {t1} and {t2} to be assignment compatible")


def check_type(found: Type, expected: Type):
    if found != expected:
        raise SemanticError(f"Found `{found}' but expected {expected}")


def check_list_type(found: Type) -> Type:
    if not isinstance(found, ListType):
        raise SemanticError(f"Found `{found}' but expected a list type")
    else:
        return found.elem_type


@dataclass
class FunctionInfo:
    func_type: FunctionType
    params: List[str]
    nested_defs: List[Tuple[str, Type]]

    def __post_init__(self):
        if len(self.func_type.inputs) != len(self.params):
            raise Exception(
                f"Expected same number of input types and parameter names")


LocalEnvironment = Dict[str, Union[Type, FunctionInfo]]


def type_checking(_: MLContext, module: ModuleOp) -> ModuleOp:
    o = build_env(module)
    r = bottom_type

    program = module.ops[0]
    assert isinstance(program, choco_ast.Program)
    if len(program.defs.ops) >= 1:
        check_stmt_or_def_list(o, r, program.defs.ops)
    if len(program.stmts.ops) >= 1:
        check_stmt_or_def_list(o, r, program.stmts.ops)

    return module


# Build local environments
def build_env(module: ModuleOp) -> LocalEnvironment:
    o: LocalEnvironment = {
        "len": FunctionInfo(FunctionType([object_type], int_type), ["arg"],
                            []),
        "print": FunctionInfo(FunctionType([object_type], none_type), ["arg"],
                              []),
        "input": FunctionInfo(FunctionType([], str_type), [], [])
    }

    @dataclass
    class BuildEnvVisitor(Visitor):
        o: LocalEnvironment

        def visit_typed_var(self, typed_var: choco_ast.TypedVar):
            name, type = typed_var.var_name.data, Type.from_op(  #type: ignore
                typed_var.type.op)
            self.o.update({name: type})

        def traverse_func_def(self, func_def: choco_ast.FuncDef):
            f: str = func_def.func_name.data  #type: ignore
            # collect function parameter names and types
            xs: List[str] = []
            ts: List[Type] = []
            for op in func_def.params.ops:
                assert isinstance(op, choco_ast.TypedVar)
                name, type = op.var_name.data, Type.from_op(  #type: ignore
                    op.type.op)
                xs.append(name)
                ts.append(type)
            # collect return type
            t = Type.from_op(func_def.return_type.op) \
                if (len(func_def.return_type.ops) == 1) else none_type
            # collect nested variable definitions
            body_visitor = BuildEnvVisitor({})
            for op in func_def.func_body.ops:
                body_visitor.traverse(op)
            vs: List[Tuple[str, Type]] = []
            for var_name, var_type in body_visitor.o.items():
                assert isinstance(var_type, Type)
                vs.append((var_name, var_type))

            o.update({f: FunctionInfo(FunctionType(ts, t), xs, vs)})

    BuildEnvVisitor(o).traverse(module)

    return o


# Dispatch to typing rules to decide which rule to invoke


def check_stmt_or_def_list(o: LocalEnvironment, r: Type, ops: List[Operation]):
    return stmt_def_list_rule(o, r, ops)


def check_stmt_or_def(o: LocalEnvironment, r: Type, op: Operation):
    if isinstance(op, choco_ast.VarDef):
        typed_var = op.typed_var.op
        assert isinstance(typed_var, choco_ast.TypedVar)
        vname = typed_var.var_name.data  #type: ignore
        e1 = op.literal.op
        return var_init_rule(o, r, vname, e1)
    elif isinstance(op, choco_ast.Assign):
        target_expr = op.target.op
        value_expr = op.value.op
        if isinstance(value_expr, choco_ast.Assign):
            return multi_assign_stmt(o, r, op)
        elif isinstance(target_expr, choco_ast.IndexExpr):
            return list_assign_stmt_rule(o, r, target_expr, value_expr)
        else:
            return var_assign_stmt_rule(o, r, target_expr, value_expr)
        # raise Exception("Support for choco_ast.Assign not implemented yet")
    elif isinstance(op, choco_ast.Pass):
        return pass_rule(o, r)
        # raise Exception("Support for choco_ast.Pass not implemented yet")
    elif isinstance(op, choco_ast.Return):
        raise SemanticError()
        # raise Exception("Support for choco_ast.Return not implemented yet")
    elif isinstance(op, choco_ast.If):
        cond = op.cond.op
        then = op.then.ops
        orelse = op.orelse.ops
        return if_else_rule(o, r, cond, then, orelse)
        # raise Exception("Support for choco_ast.If not implemented yet")
    elif isinstance(op, choco_ast.While):
        cond = op.cond.op
        body = op.body.ops
        return while_rule(o, r, cond, body)
        # raise Exception("Support for choco_ast.While not implemented yet")
    elif isinstance(op, choco_ast.For):
        iter_name = op.iter_name.data
        iter_expr = op.iter.op
        body = op.body.ops
        if isinstance(iter_expr, choco_ast.ListExpr):
            return for_list_rule(o, r, iter_name, iter_expr, body)
        else:
            return for_str_rule(o, r, iter_expr, body)
        # raise Exception("Support for choco_ast.For not implemented yet")
    elif isinstance(op, choco_ast.GlobalDecl):
        raise Exception("Support for choco_ast.GlobalDecl not implemented yet")
    elif isinstance(op, choco_ast.NonLocalDecl):
        raise Exception(
            "Support for choco_ast.NonLocalDecl not implemented yet")
    elif isinstance(op, choco_ast.FuncDef):
        return func_def_rule(o, r, op)
        # raise Exception("Support for choco_ast.FuncDef not implemented yet")
    else:
        return expr_stmt_rule(o, r, op)

def check_expr(o: LocalEnvironment, r: Type, op: Operation) -> Type:
    t: Optional[Type] = None
    if isinstance(op, choco_ast.Literal):
        t = literal_rules(o, r, op)
    elif isinstance(op, choco_ast.UnaryExpr):
        op_name = op.op.data  #type: ignore
        e = op.value.op
        if op_name == "-":
            t = negate_rule(o, r, e)
        elif op_name == "not":
            t = not_rule(o, r, e)
        else:
            raise Exception(
                "Support for some choco_ast.UnaryExpr not implemented yet")
    elif isinstance(op, choco_ast.BinaryExpr):
        op_name = op.op.data
        lhs = op.lhs.op
        rhs = op.rhs.op
        if op_name == 'and':
            t = and_rule(o, r, lhs, rhs)
        elif op_name == 'is':
            t = is_rule(o, r, lhs, rhs)
        elif op_name == 'or':
            t = or_rule(o, r, lhs, rhs)
        elif op_name in ['>', '<', '==', '!=','>=', '<=']:
            tl = check_expr(o, r, lhs)
            tr = check_expr(o, r, rhs)
            if tl == str_type:
                t = str_compare_rule(o, r, tr)
            elif tl == int_type:
                t = int_compare_rule(o, r, tr)
            elif tl == bool_type:
                t = bool_compare_rule(o, r, tr)
            else:
                raise SemanticError("invalid object comarsion")
        elif op_name in ['+', '-', '*', '//', '%']:
            type_left = check_expr(o, r, lhs)
            type_right = check_expr(o, r, rhs)
            if type_left == str_type:
                t = str_concat_rule(o, r, type_right)
            elif type_left == int_type:
                t = arith_rule(o, r, type_right)
            elif isinstance(type_left, ListType):
                t = list_concat_rule(o, r, type_left, type_right)
        else:
            raise Exception("Support for choco_ast.BinaryExpr not implemented yet")
    elif isinstance(op, choco_ast.ExprName):
        t = var_read_rule(o, r, op.id.data)  #type: ignore
    elif isinstance(op, choco_ast.IfExpr):
        e0 = op.cond.op
        e1 = op.then.op
        e2 = op.or_else.op
        t = cond_rule(o, r, e0, e1, e2)
    elif isinstance(op, choco_ast.IndexExpr):
        value = op.value.op
        index = op.index.op

        if isinstance(value, choco_ast.ExprName):
            id = value.id.data
            if o[id] == str_type:
                t = str_select_rule(o, r, value, index)
            else:
                t = list_select_rule(o, r, value, index)
        elif isinstance(value, choco_ast.ListExpr):
            t = list_select_rule(o, r, value, index)
        elif isinstance(value, choco_ast.Literal):
            if isinstance(value.value, choco_ast.StringAttr):
                t = str_select_rule(o, r, value, index)
            else:
                raise SemanticError(f"Found `{type(value.value)}' but expected {str_type}")
        else:
            SemanticError(
                f"Unkown {type(value)} where {ListType} or {str_type} expected")

        # raise Exception("Support for choco_ast.IndexExpr not implemented yet")
    elif isinstance(op, choco_ast.ListExpr):
        t = list_display_rule(o, r, op)
        # raise Exception("Support for choco_ast.ListExpr not implemented yet")
    elif isinstance(op, choco_ast.CallExpr):

        t = invoke_rule(o, r, op)

        # raise Exception("Support for choco_ast.CallExpr not implemented yet")

    assert isinstance(t, Type)

    op.attributes["type_hint"] = to_attribute(t)

    return t


# Typing rules, each typing rule is implemented by a corresponding function


# [VAR-READ] rule
# O, R |- id: T
def var_read_rule(o: LocalEnvironment, r: Type, id: str) -> Type:
    # O(id) = T, where T is not a function type.
    t = o.get(id)
    if t is None:
        raise SemanticError(f"Unknown identifier {id} used")
    if isinstance(t, FunctionInfo):
        raise SemanticError(
            f"Function identifier `{id}' used where value identifier expected")
    return t


# [VAR-INIT] rule
# O, R |- id: T = e1
def var_init_rule(o: LocalEnvironment, r: Type, id: str, e1: Operation):
    # O(id) = T
    t = o[id]
    if isinstance(t, FunctionInfo):
        raise SemanticError(
            f"Function identifier `{id}' used where value identifier expected")
    # O, R |- e1: T1
    t1 = check_expr(o, r, e1)
    # T1 ≤a T
    check_assignment_compatibility(t1, t)


# [VAR-ASSIGN-STMT]
# O, R |- id = e1
def var_assign_stmt_rule(o: LocalEnvironment, r: Type, id: Operation, e1: Operation):
    # O(id) = T
    t = check_expr(o, r, id)

    # O, R |- e1: T1
    t1 = check_expr(o, r, e1)

    # T1 ≤a T
    check_assignment_compatibility(t1, t)

# [STMT-DEF-LIST] rule
# O, R |- s1 NEWLINE s2 NEWLINE . . . sn NEWLINE
def stmt_def_list_rule(o: LocalEnvironment, r: Type, sns: List[Operation]):
    # n >= 1
    assert len(sns) >= 1
    for sn in sns:
        # O, R |- sn
        check_stmt_or_def(o, r, sn)


# [PASS] rule
# O, R |- pass
def pass_rule(o: LocalEnvironment, r: Type):
    pass


# [EXPR-STMT] rule
# O, R |- e
def expr_stmt_rule(o: LocalEnvironment, r: Type, e: Operation):
    # O, R |- e: T
    _t = check_expr(o, r, e)


# noinspection PySimplifyBooleanCheck
# O, R |- lit: T
def literal_rules(o: LocalEnvironment, r: Type,
                  lit: choco_ast.Literal) -> Type:
    if isinstance(lit.value, choco_ast.BoolAttr):
        # [BOOL-FALSE] rule
        if lit.value.data == False:
            return bool_type
        # [BOOL-TRUE] rule
        if lit.value.data == True:
            return bool_type
    # [NONE] rule
    if isinstance(lit.value, choco_ast.NoneAttr):
        return none_type
    # [INT] rule
    if isinstance(lit.value, choco_ast.IntegerAttr):
        return int_type
    # [STR] rule
    if isinstance(lit.value, choco_ast.StringAttr):
        return str_type
    raise Exception(f"Could not type check literal `{lit}'")


# [NEGATE] rule
# O, R, |- - e: int
def negate_rule(o: LocalEnvironment, r: Type, e: Operation) -> Type:
    # O, R, |- e: int
    check_type(check_expr(o, r, e), expected=int_type)
    return int_type


# [ARITH] rule
# O, R |- e1 op e2 : int
def arith_rule(o: LocalEnvironment, r: Type, t: Type) -> Type:
    check_type(t, expected=int_type)
    return int_type

# [INT-COMPARE] rule
# O, R |- e1 cmp_op e2 : bool
def int_compare_rule(o: LocalEnvironment, r: Type, t: Type) -> Type:
    check_type(t, expected=int_type)
    return bool_type

# [BOOL-COMPARE] rule
# O, R |- e1 cmp_op e2 : bool
def bool_compare_rule(o: LocalEnvironment, r: Type, t: Type) -> Type:
    check_type(t, expected=bool_type)
    return bool_type

# [AND] rule
# O, R |- e1 and e2 : bool
def and_rule(o: LocalEnvironment, r: Type, lhs: Operation, rhs: Operation) -> Type:
    check_type(check_expr(o, r, lhs), expected=bool_type)
    check_type(check_expr(o, r, rhs), expected=bool_type)
    return bool_type

# [OR] rule
# O, R |- e1 or e2 : bool
def or_rule(o: LocalEnvironment, r: Type, lhs: Operation, rhs: Operation) -> Type:
    check_type(check_expr(o, r, lhs), expected=bool_type)
    check_type(check_expr(o, r, rhs), expected=bool_type)
    return bool_type

# [NOT] rule
# O, R |- not e : bool
def not_rule(o: LocalEnvironment, r: Type, e: Operation) -> Type:
    check_type(check_expr(o, r, e), expected=bool_type)
    return bool_type


# [COND] rule
# O, R |- e1 if e0 else e2 : T1 |_| T2
def cond_rule(o: LocalEnvironment, r: Type, e0: Operation, e1: Operation, e2: Operation) -> Type:
    # O, R |- e0: bool
    check_type(check_expr(o, r, e0), expected=bool_type)
    # O, R |- e1: t1
    t1 = check_expr(o, r, e1)
    # O, R |- e2: t2
    t2 = check_expr(o, r, e2)

    return join(t1, t2)


# [STR-COMPARE] rule
# O, R |- e1 cmp_op e2 : bool
def str_compare_rule(o: LocalEnvironment, r: Type, t: Type) -> Type:
    check_type(t, expected=str_type)
    return bool_type

# [STR-CONCAT]
# O, R |- e1 + e2 : str
def str_concat_rule(o: LocalEnvironment, r: Type, t: Type) -> Type:
    check_type(t, expected = str_type)
    return str_type

# [STR-SELECT]
# O, R |- e1[e2] : str
def str_select_rule(o: LocalEnvironment, r: Type, e1: Operation, e2: Operation) -> Type:
    t1 = check_expr(o, r, e1)
    t2 = check_expr(o, r, e2)
    check_type(t1, expected=str_type)
    check_type(t2, expected=int_type)
    return str_type

# [IS]
# O, R |- e1 is e2 : bool
def is_rule(o: LocalEnvironment, r: Type, e1: Operation, e2: Operation) -> Type:
    t = check_expr(o, r, e1)
    t1 = check_expr(o, r, e2)

    if isinstance(e1, choco_ast.ListExpr) and isinstance(e2, choco_ast.ListExpr):
        return bool_type
    else:
        check_type(t, expected=t1)
        return bool_type

# [LIST-DISPLAY]
# O, R |- [e1, e2, ..., en] : [T]
def list_display_rule(o: LocalEnvironment, r: Type, op_list: List[Operation]) -> Type:
    op = op_list.elems.blocks[0].ops

    if not op:
        return nil_rule(o, r)
    else:
        t = check_expr(o, r, op[0])
        for e in op[1:]:
            t1 = check_expr(o, r, e)
            t = join(t, t1)
        return ListType(t)

# [NIL]
# O, R |- [] : <Empty>
def nil_rule(o: LocalEnvironment, r: Type) -> Type:
    return empty_type

# [LIST-CONCAT]
# O, R |- e1 + e2 : [T]
def list_concat_rule(o: LocalEnvironment, r: Type, t1: Type, t2:Type) -> Type:
    check_type(ListType(check_list_type(t1)), expected=ListType(check_list_type(t2)))
    return ListType(check_list_type(t1))

# [LIST-SELECT]
# O, R |- e1[e2] : T
def list_select_rule(o: LocalEnvironment, r: Type, e1: Operation, e2: Operation) -> Type:
    t1 = check_expr(o, r, e1)
    t2 = check_list_type(t1)
    check_type(check_expr(o, r, e2), expected=int_type)
    return t2

# [LIST-ASSIGN-STMT]
# O, R |- e1[e2] = e3
def list_assign_stmt_rule(o: LocalEnvironment, r: Type, e1: Operation, e2: Operation):
    t1 = check_expr(o, r, e1)
    t2 = check_expr(o, r, e2)
    t = check_list_type(check_expr(o, r, e1.value.op))
    check_assignment_compatibility(t, t2)

# [MULTI-ASSIGN-STMT]
# O,R |- e1 = e2 = ... = en = e0
def multi_assign_stmt(o: LocalEnvironment, r: Type, op: Operation):
    t = check_expr(o, r, op.target.op)
    op = op.value.op
    while isinstance(op, choco_ast.Assign):
        target = op.target.op
        t1 = check_expr(o, r, target)
        op = op.value.op
        check_assignment_compatibility(t, t1)

    t2 = check_expr(o, r, op)
    check_assignment_compatibility(t2, t)

# [INVOKE]
# O, R |- f(e1, e2, ..., en): T0
def invoke_rule(o: LocalEnvironment, r: Type, op: Operation) -> Type:
    func = op.func
    args = op.args
    type_hint = op.type_hint

    for typed_var in args.ops:
        check_expr(o, r, typed_var)

    t = o[func.data].func_type.output

    return t

# [RETURN-e] rule
# O, R |- return e
# def return_e_rule(o: LocalEnvironment, r: Type, ???):
#     ???

# [RETURN] rule
# O, R |- return
# def return_rule(o: LocalEnvironment, r: Type, ???):
#     ???

# [IF-ELSE]
# O, R |- if e: b1 else: b2
def if_else_rule(o: LocalEnvironment, r: Type, e: Operation, b1: List[Operation], b2: List[Operation]):
    t = check_expr(o, r, e)
    check_type(t, expected=bool_type)
    if b1:
        for op1 in b1:
            check_stmt_or_def(o, r, op1)

    if b2:
        for op2 in b2:
            check_stmt_or_def(o, r, op2)

# [WHILE]
# O, R |- while e: b
def while_rule(o: LocalEnvironment, r: Type, e: Operation, b: List[Operation]):
    t = check_expr(o, r, e)
    check_type(t, expected=bool_type)
    for op in b:
        check_stmt_or_def(o, r, op)

# [FOR-STR]
# O, R |- for id in e: b
def for_str_rule(o: LocalEnvironment, r: Type, e: Operation, b: List[Operation]):
    t = check_expr(o, r, e)
    check_type(t, expected=str_type)
    for op in b:
        check_stmt_or_def(o, r, op)

# [FOR-LIST]
# O, R |- for id in e: b
def for_list_rule(o: LocalEnvironment, r: Type, id: str, e: Operation, b: List[Operation]):
    t = check_expr(o, r, e)
    check_type(t, expected=ListType(o[id]))
    for op in b:
        check_stmt_or_def(o, r, op)

# [FUNC-DEF] rule
# O, R |- def f(x1:T1, ... , xn:Tn)  [[-> T0]]? :b
def func_def_rule(o: LocalEnvironment, r: Type, func_op: Operation):
    func_name = func_op.func_name
    params = func_op.params
    return_type = func_op.return_type
    func_body = func_op.func_body

    if len(func_body.ops) == 1:
        if isinstance(func_body.op, choco_ast.Return):
            if func_body.op.value.ops:
                for op in func_body.op.value.ops:
                    t = check_expr(o, r, op)
        elif isinstance(func_body.op, choco_ast.Assign):
            typed_params = list(zip(o[func_name.data].params, o[func_name.data].func_type.inputs))
            for var_name, var_type in typed_params:
                o[var_name] = var_type
            t1 = check_expr(o, r, func_body.op.target.op)
            t2 = check_expr(o, r, func_body.op.value.op)
    else:
        for op_expr in func_body.ops:
            if isinstance(op_expr, choco_ast.GlobalDecl):
                continue
            elif isinstance(op_expr, choco_ast.Assign):
                typed_params = list(zip(o[func_name.data].params, o[func_name.data].func_type.inputs))
                for var_name, var_type in typed_params:
                    o[var_name] = var_type
                t1 = check_expr(o, r, op_expr.target.op)
                t2 = check_expr(o, r, op_expr.value.op)
