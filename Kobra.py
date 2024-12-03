#Code is semi aided by GeeksForGeeks (included in resources for this project) and CodePulse (also in class resources)


#CONSTANT


DIGITS = '01234567890'

#ERRORS


class Error:
    def __init__(self, pos_start, pos_end, error_name, details):
        self.pos_start = pos_start
        self.pos_end = pos_end
        self.error_name = error_name
        self.details = details

    def as_string(self):
        result = f'{self.error_name}: {self.details}'
        result += f' File {self.pos_start.fn}, line {self.pos_start.ln + 1}'
        return result

class IllegalCharError(Error):
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end, 'Illegal Character', details)

class InvalidSyntaxError(Error):
    def __init__(self, pos_start, pos_end, details= ''):
        super().__init__(pos_start, pos_end, 'Invalid syntax', details)

class RTError(Error):
    def __init__(self, pos_start, pos_end, details= ''):
        super().__init__(pos_start, pos_end, 'Runtime Error', details)




#POSITION


class Position:
    def __init__(self, idx, ln, col, fn, ftxt):
        self.idx = idx
        self.ln = ln
        self.col = col
        self.fn = fn
        self.ftxt = ftxt

    def advance(self, current_char=None):
        self.idx += 1
        self.col += 1

        if current_char == '\n':
            self.ln += 1
            self.col = 0
        return self
    def copy(self):
        return Position(self.idx, self.ln, self.col, self.fn, self.ftxt)

#TOKENS

TT_INT = 'INT'
TT_FLOAT = 'FLOAT'
TT_PLUS = 'PLUS'
TT_MINUS = 'MINUS'
TT_MUL = 'MUL'
TT_DIV = 'DIV'
TT_LPAREN = 'LPAREN'
TT_RPAREN = 'RPAREN'
TT_EOF = 'EOF'
TT_STRING = 'STRING'
TT_IDENTIFIER = 'IDENTIFIER'
TT_ASSIGN = 'ASSIGN'
TT_EQ = 'EQ'


class Token:
    def __init__(self, type_, value=None, pos_start = None, pos_end = None):
        self.type = type_
        self.value = value

        if pos_start:
            self.pos_start = pos_start.copy()
            self.pos_end = pos_start.copy()
            self.pos_end.advance()
        
        if pos_end:
            self.pos_end = pos_end

    def __repr__(self):
        if self.value: return f'{self.type}:{self.value}'
        return f'{self.type}'
    

#LEXER

class Lexer:
    def __init__(self, fn, text):
        self.fn = fn
        self.text = text
        self.pos = Position(-1, 0, -1, fn, text)
        self.current_char = None
        self.advance()

    def advance(self):
        self.pos.advance(self.current_char)
        self.current_char = self.text[self.pos.idx] if self.pos.idx < len(self.text) else None

    def make_tokens(self):
        tokens = []

        while self.current_char != None:
            if self.current_char in ' \t':
                self.advance()
            elif self.current_char in DIGITS:
                tokens.append(self.make_number())
            elif self.current_char == "'":
                tokens.append(self.make_string("'"))
            elif self.current_char == '"':
                tokens.append(self.make_string('"'))
            elif self.current_char == '+':
                tokens.append(Token(TT_PLUS, pos_start=self.pos))
                self.advance()
            elif self.current_char == '-':
                tokens.append(Token(TT_MINUS, pos_start=self.pos))
                self.advance()
            elif self.current_char == '*':
                tokens.append(Token(TT_MUL, pos_start=self.pos))
                self.advance()
            elif self.current_char == '/':
                tokens.append(Token(TT_DIV, pos_start=self.pos))
                self.advance()
            elif self.current_char == '(':
                tokens.append(Token(TT_LPAREN, pos_start=self.pos))
                self.advance()
            elif self.current_char == ')':
                tokens.append(Token(TT_RPAREN, pos_start=self.pos))
                self.advance()
            elif self.current_char == '=':
                tokens.append(Token(TT_EQ, pos_start=self.pos))
                self.advance()
            elif self.current_char.isalpha() or self.current_char == '_':
                tokens.append(self.make_identifier())
            else:
                pos_start = self.pos.copy()
                char = self.current_char
                self.advance()
                return [], IllegalCharError(pos_start, self.pos, "'" + char + "'")
        
        tokens.append(Token(TT_EOF, pos_start=self.pos))
        return tokens, None

    def make_string(self, quote_type):
        string = ''
        pos_start = self.pos.copy()

        self.advance()  # Skip the initial quote

        while self.current_char != None and self.current_char != quote_type:
            string += self.current_char
            self.advance()

        if self.current_char == quote_type:
            self.advance()  # Skip the closing quote
            return Token(TT_STRING, string, pos_start, self.pos)
        else:
            return [], IllegalCharError(pos_start, self.pos, f"Expected '{quote_type}' to close string")
    
    def make_number(self):
        num_str = ''
        dot_count = 0
        pos_start = self.pos.copy()

        while self.current_char !=  None and self.current_char in DIGITS + '.':
            if self.current_char == '.':
                if dot_count == 1: break
                dot_count += 1
                num_str += '.'
            else:
                num_str += self.current_char
                self.advance()

        if dot_count == 0:
            return Token(TT_INT, int(num_str), pos_start, self.pos)
        else:
            return Token(TT_FLOAT, float(num_str), pos_start, self.pos)
    
    def make_identifier(self):
        identifier = ''
        pos_start = self.pos.copy()

        while self.current_char != None and (self.current_char.isalnum() or self.current_char == '_'):
            identifier += self.current_char
            self.advance()

        return Token(TT_IDENTIFIER, identifier, pos_start, self.pos)





#NODES


class NumberNode:
    def __init__(self, tok):
        self.tok = tok

        self.pos_start = self.tok.pos_start
        self.pos_end = self.tok.pos_end
    def __repr__(self):
        return f'{self.tok.type}:{self.tok.value}'
        
class StringNode:
    def __init__(self, tok):
        self.tok = tok
        self.pos_start = self.tok.pos_start
        self.pos_end = self.tok.pos_end

    def __repr__(self):
        return f'{self.tok.type}:{self.tok.value}'

class BinOpNode:
    def __init__(self, left_node, op_tok, right_node):
        self.left_node = left_node
        self.op_tok = op_tok
        self.right_node = right_node

        self.pos_start = self.left_node.pos_start
        self.pos_end = self.right_node.pos_end
    def __repr__(self):
        return f'({self.left_node}, {self.op_tok.type}, {self.right_node})'
        
class UnaryOpNode:
    def __init__(self, op_tok, node):
        self.op_tok = op_tok
        self.node = node

        self.pos_start = self.op_tok.pos_start
        self.pos_end = node.pos_end
    
    def __repr__(self):
        return f'({self.op_tok}, {self.node})'


#PARSE RESULT


class ParseResult:
    def __init__(self):
        self.error = None
        self.node = None
    
    def register(self, res):
        if isinstance(res, ParseResult):
            if res.error: self.error = res.error
            return res.node
        return res
    
    def success(self, node):
        self.node = node
        return self

    def failure(self, error):
        self.error = error
        return self



#PARSER


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.tok_idx = -1
        self.advance()

    def advance(self):
        self.tok_idx += 1
        if self.tok_idx < len(self.tokens):
            self.current_tok = self.tokens[self.tok_idx]
        return self.current_tok

    def parse(self):
        res = self.statement()  # Try to parse an assignment or expression
        if not res.error and self.current_tok.type != TT_EOF:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected '+', '-', '*', or '/'"
            ))
        return res


    def factor(self):
        res = ParseResult()
        tok = self.current_tok

        if tok.type in (TT_PLUS, TT_MINUS):
            res.register(self.advance())
            factor = res.register(self.factor())
            if res.error: return res
            return res.success(UnaryOpNode(tok, factor))

        elif tok.type in (TT_INT, TT_FLOAT):
            res.register(self.advance())
            return res.success(NumberNode(tok))

        elif tok.type == TT_STRING:
            res.register(self.advance())
            return res.success(StringNode(tok))

        elif tok.type == TT_LPAREN:
            res.register(self.advance())
            expr = res.register(self.expr())
            if res.error: return res
            if self.current_tok.type == TT_RPAREN:
                res.register(self.advance())
                return res.success(expr)
            else:
                return res.failure(InvalidSyntaxError(
                        self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected ')'"
                ))
        elif tok.type == TT_IDENTIFIER:
            var_name = tok.value
            res.register(self.advance())  # Move past the identifier
            return res.success(VarAccessNode(var_name, tok.pos_start, tok.pos_end))

        elif tok.type == TT_INT or tok.type == TT_FLOAT:
            res.register(self.advance())
            return res.success(NumberNode(tok))

        return res.failure(InvalidSyntaxError(
            tok.pos_start, tok.pos_end,
            "expected int, float or string"
        ))


    def term(self):
        return self.bin_op(self.factor, (TT_MUL, TT_DIV))
    
    def expr(self):
        return self.bin_op(self.term, (TT_PLUS, TT_MINUS))

    def bin_op(self, func, ops):
        res = ParseResult()
        left = res.register(func())
        if res.error: return res

        while self.current_tok and self.current_tok.type in ops:
            op_tok = self.current_tok
            res.register(self.advance())
            right = res.register(func())
            if res.error: return res
            left = BinOpNode(left, op_tok, right)

        return res.success(left)
    
    def statement(self):
        res = ParseResult()

        if self.current_tok.type == TT_IDENTIFIER:
            var_name = self.current_tok.value
            res.register(self.advance())  # Move past the identifier

            if self.current_tok.type == TT_EQ:
                res.register(self.advance())  # Move past the '='
                expr = res.register(self.expr())
                if res.error: return res
                return res.success(VarAssignNode(var_name, expr))

    
# parse as expresssion
        expr = res.register(self.expr())
        if res.error:
            return res.failure(InvalidSyntaxError(self.current_tok.pos_start, self.current_tok.pos_end, 'Expected an assignment or expression'))
        return res.success(expr)




#VARIABLE ASSIGNMENT

class VarAssignNode:
    def __init__(self, var_name, expr):
        self.var_name = var_name
        self.expr = expr
        self.pos_start = expr.pos_start
        self.pos_end = expr.pos_end

    def __repr__(self):
        return f'({self.var_name} = {self.expr})'



#RUNTIME RESULT

class RTResult:
    def __init__(self):
        self.value = None
        self.error = None
    def register(self, res):
        if res.error: self.error = res.error
        return res.value

    def success(self, value):
        self.value = value
        return self
    
    def failure(self, error):
        self.error = error
        return self


#VALUES

class Number:
    def __init__(self, value):
        self.value = value
        self.set_pos()

    def set_pos(self, pos_start=None, pos_end=None):
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self
    def added_to(self, other):
        if isinstance(other, Number):
            return Number(self.value + other.value), None
    def subbed_by(self, other):
        if isinstance(other, Number):
            return Number(self.value - other.value), None
    def multed_by(self, other):
        if isinstance(other, Number):
            return Number(self.value * other.value), None
    def divided_by(self, other):
        if isinstance(other, Number):
            if other.value == 0:
                return None, RTError(
                    other.pos_start, other.pos_end,
                    'Division by 0'
                )
            return Number(self.value / other.value), None
    def __repr__(self):
        return str(self.value)


#INTERPRETER

class Interpreter:
    def __init__(self):
        self.symbol_table = SymbolTable()

    def visit_VarAssignNode(self, node):
        value = self.visit(node.expr)
        if value.error:
            return value

        self.symbol_table.set(node.var_name, value.value)
        return RTResult().success(value)

    def visit_VarAccessNode(self, node):
        value = self.symbol_table.get(node.var_name)
        if value is None:
            return RTResult().failure(RTError(node.pos_start, node.pos_end, f"Undefined variable '{node.var_name}'"))
        return RTResult().success(value)


    def visit(self, node):
        method_name = f'visit_{node.__class__.__name__}'
        method = getattr(self, method_name, self.no_visit_method)
        return method(node)

    def no_visit_method(self, node):
        raise Exception(f'No visit_{node.__class___.__name__} method defined')

    def visit_NumberNode(self, node):
        return RTResult().success(Number(node.tok.value).set_pos(node.pos_start, node.pos_end))

    def visit_StringNode(self, node):
        return RTResult().success(String(node.tok.value).set_pos(node.pos_start, node.pos_end))

    def visit_BinOpNode(self, node):
        res = RTResult()
        left = res.register(self.visit(node.left_node))
        if res.error: return res
        right = res.register(self.visit(node.right_node))
        if res.error: return res

        if node.op_tok.type == TT_PLUS:
            if isinstance(left, String) and isinstance(right, String):
                return res.success(String(left.value + right.value).set_pos(node.pos_start, node.pos_end))
            elif isinstance(left, Number) and isinstance(right, Number):
                return res.success(Number(left.value + right.value).set_pos(node.pos_start, node.pos_end))
            else:
                return res.failure(InvalidSyntaxError(
                    node.pos_start, node.pos_end,
                    f"Invalid operation '{left}' + '{right}'"
                ))
        elif node.op_tok.type == TT_MINUS:
            result, error = left.subbed_by(right)
        elif node.op_tok.type == TT_MUL:
            result, error = left.multed_by(right)
        elif node.op_tok.type == TT_DIV:
            result, error = left.divided_by(right)

        if error:
            return res.failure(error)
        else:
                return res.success(result.set_pos(node.pos_start, node.pos_end))


    def visit_UnaryOpNode(self, node):
        res = RTResult()
        number = res.register(self.visit(node.node))
        if res.error: return res

        error = None

        if node.op_tok.type == TT_MINUS:
            number, error = number.multed_by(Number(-1))

        if error:
            return res.failure(error)
        else:
            return res.success(number.set_pos(node.pos_start, node.pos_end))


class String:
    def __init__(self, value):
        self.value = value
        self.set_pos()

    def set_pos(self, pos_start=None, pos_end=None):
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self

    def __repr__(self):
        return f'"{self.value}"'


#SYMBOL TABLE

class SymbolTable:
    def __init__(self):
        self.symbols = {}

    def get(self, var_name):
        if var_name in self.symbols:
            return self.symbols[var_name]
        return None

    def set(self, var_name, value):
        self.symbols[var_name] = value


#VARIABLE ACCESS

class VarAccessNode:
    def __init__(self, var_name, pos_start, pos_end):
        self.var_name = var_name
        self.pos_start = pos_start
        self.pos_end = pos_end

    def __repr__(self):
        return self.var_name




#RUN

def run(fn, text, interpreter):
    lexer = Lexer(fn, text)
    tokens, error = lexer.make_tokens()
    if error: return None, error
    
    
    parser = Parser(tokens)
    ast = parser.parse()
    if ast.error: return None, ast.error

    
    result = interpreter.visit(ast.node)

    return result.value, result.error
