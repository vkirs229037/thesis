from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Dict, Tuple, Any, NoReturn
from graph import Graph, Vertex, GraphKind
import igraph as ig
import algs

class TType(Enum):
    Id = auto()
    Arrow = auto()
    LBrace = auto()
    RBrace = auto()
    LBracket = auto()
    RBracket = auto()
    Semicolon = auto()
    Colon = auto()
    String = auto()
    Numlit = auto()
    Eof = auto()


@dataclass
class Token:
    type: TType
    value: str
    line: int
    col: int


class LexError(Exception):
    pass

class Lexer:
    def __init__(self, content: str):
        self.content = content
        self.cur = 0
        self.is_reading_label = False
        self.in_comment = False
        self.tokens: List[Token] = []
        self.col = 0
        self.line = 0

    def peek(self, forward: int = 0) -> str | None:
        if self.cur + forward < len(self.content):
            return self.content[self.cur + forward]
        return None
    
    def consume(self) -> str | None:
        result = None
        if self.cur < len(self.content):
            result = self.content[self.cur]
        self.cur += 1
        return result
    
    def take_id(self) -> str:
        result: List[str] = []
        while (c := self.peek()) is not None and c.isalnum(): 
            # здесь self.consume() не будет None
            result.append(self.consume()) # type: ignore
        self.cur -= 1
        return "".join(result)
    
    def take_string(self) -> str:
        result: List[str] = []
        while (c := self.peek()) != "\"": 
            if c is None:
                raise LexError(f"{self.line}:{self.col} Ошибка: ожидалась \", встречен конец файла")
            # здесь self.consume() не будет None
            result.append(self.consume()) # type: ignore
        return "".join(result)
        
    def get_token(self) -> Token:
        token: Token | None = None
        c = self.content[self.cur]

        match c:
            case c if c.isalnum():
                token = self.parse_id()
            case "#":
                token = self.parse_numlit()
            case ";":
                token = Token(TType.Semicolon, ";", self.line, self.col)
            case ":":
                token = Token(TType.Colon, ":", self.line, self.col)
            case "]":
                token = Token(TType.RBracket, "]", self.line, self.col)
            case "[":
                token = Token(TType.LBracket, "[", self.line, self.col)
            case "}":
                token = Token(TType.RBrace, "}", self.line, self.col)
            case "{":
                token = Token(TType.LBrace, "{", self.line, self.col)
            case ">":
                token = Token(TType.Arrow, ">", self.line, self.col)
            case "\"":
                self.cur += 1
                self.col += 1
                token = Token(TType.String, self.take_string(), self.line, self.col)
            case _:
                raise LexError(f"{self.line}:{self.col} Ошибка: неизвестный символ {c}")
        self.cur += 1
        return token
    
    def parse_id(self) -> Token:
        id_text = self.take_id()
        return Token(TType.Id, id_text, self.line, self.col)
    
    def parse_numlit(self) -> Token:
        self.cur += 1
        self.col += 1
        result: List[str] = []
        while (c := self.peek()) is not None and self.is_numeric(c):
            # здесь точно self.consume() не будет None
            result.append(self.consume()) # type: ignore
        self.cur -= 1
        str_value = "".join(result)
        try:
            _ = int(str_value)
        except ValueError:
            raise LexError(f"{self.line}:{self.col} Ошибка: неверно записанное десятичное число")
        return Token(TType.Numlit, "".join(result), self.line, self.col)

    def is_numeric(self, c: str):
        return c.isdigit() or c == "-"
    
    def lex(self) -> List[Token]:
        while (c := self.peek()) is not None:
            if self.in_comment:
                if c == "\n":
                    self.consume()
                    self.line += 1
                    self.col = 0
                    self.in_comment = False
                else:
                    self.consume()
                    self.col += 1
            else:
                if c == "\n":
                    self.consume()
                    self.line += 1
                    self.col = 0
                elif c.isspace():
                    self.col += 1
                    self.consume()
                elif c == "/":
                    if self.peek(1) == "/":
                        self.in_comment = True
                        self.col += 2
                        self.consume()
                        self.consume()
                else:
                    token = self.get_token()
                    self.tokens.append(token)
                    self.col += len(token.value)
        self.tokens.append(Token(TType.Eof, "", self.line, self.col))
        return self.tokens


@dataclass
class Command:
    func_name: Token
    args: List[Token]

class ParseError(Exception):
    pass

class Parser:
    RESERVED_IDS = ["directed", "undirected", "vertex", "graph", "visual", "algs", "coloring", "dijkstra", "eulerness", "fleury", "floyd", "degrees", "connectivity"]
    ALGS = ["coloring", "dijkstra", "eulerness", "fleury", "floyd", "degrees", "connectivity", "strongcomps"]
    PROPS = ["palette", "edgewidth", "layout", "vertexsize"]

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.vertices: List[Vertex] = []
        self.vertices_dict: Dict[str, int] = {}
        self.cur = 0
        self.LAST_ID = -1
        self.vertex_connections: Dict[Tuple[Vertex, Vertex], int] = {}
        self.commands: List[Command] = []
        self.graph_kind: GraphKind = GraphKind.Directed
        self.last_command_line: int = -1
        self.visual = {}

    def report_parse_err(self, token: Token | None, msg: str) -> NoReturn:
        if token is None:
            raise ParseError(f"{msg}")
        else:
            raise ParseError(f"{token.line}:{token.col} {msg}")

    def last_id(self) -> int:
        self.LAST_ID += 1
        return self.LAST_ID
    
    def peek(self) -> Token | None:
        if self.cur < len(self.tokens):
            return self.tokens[self.cur]
        return None
    
    def consume(self) -> Token | None:
        result = None
        if self.cur < len(self.tokens):
            result = self.tokens[self.cur]
        self.cur += 1
        return result
    
    def parse_vertex_stmt(self):
        ident = self.consume()
        if ident is None or ident.type != TType.Id:
            self.report_parse_err(ident, "Ожидался идентификатор вершины")
        
        if ident.value in self.RESERVED_IDS:
            self.report_parse_err(ident, "Зарезервированное имя")
        
        tok = self.consume()
        if tok is None or tok.type not in [TType.Colon, TType.Semicolon]:
            self.report_parse_err(tok, "Ожидалось двоеточие или точка с запятой")
        elif tok.type == TType.Semicolon:
            vertex = Vertex(ident.value, "", self.last_id())
            self.vertices.append(vertex)
            self.vertices_dict[vertex.name] = vertex.id
            return
        elif tok.type == TType.Colon:
            label = self.consume()
            if label.type != TType.String:
                self.report_parse_err(label, "Ожидалась строка")
            
            vertex = Vertex(ident.value, label.value, self.last_id())
            self.vertices.append(vertex)
            self.vertices_dict[vertex.name] = vertex.id

            tok = self.consume()
            if tok is None or tok.type != TType.Semicolon:
                self.report_parse_err(tok, "Ожидалась точка с запятой")

    
    def parse_vertices(self):
        t = self.consume()
        if t is None or t.type != TType.LBrace:
            self.report_parse_err(t, "Ожидалась {{")

        while (t := self.peek()) is not None and (t.type != TType.RBrace and t.type != TType.Eof):
            self.parse_vertex_stmt()

        t = self.consume()
        if t is None or t.type != TType.RBrace:
            self.report_parse_err(t, "Ожидалась }}")

    def parse_graph_kind(self):
        gkind = self.consume()
        if gkind is None or gkind.type != TType.Id:
            self.report_parse_err(gkind, "Ожидался типа графа (\"directed\", \"undirected\")")

        match gkind.value:
            case "directed":
                self.graph_kind = GraphKind.Directed
            case "undirected":
                self.graph_kind = GraphKind.Undirected
            case _:
                self.report_parse_err(gkind, f"Неизвестный тип графа {gkind.value}")

        tok = self.consume()
        if tok is None or tok.type != TType.Semicolon:
            self.report_parse_err(tok, "Ожидалась точка с запятой")

    def parse_ident_conn(self) -> List[int]:
        tok = self.consume()
        if tok is None or tok.type not in [TType.Id, TType.LBracket]:
            self.report_parse_err(tok, "Ожидался идентификатор или список идентификаторов")

        if tok.type == TType.Id:
            try:
                return [self.vertices_dict[tok.value]]
            except KeyError:
                self.report_parse_err(tok, "Неизвестная вершина")
        elif tok.type == TType.LBracket:
            id_list: List[int] = []
            while (t := self.consume()) is not None and t.type == TType.Id:
                try:
                    id_v = self.vertices_dict[t.value]
                    id_list.append(id_v)
                except KeyError:
                    self.report_parse_err(tok, "Неизвестная вершина")
            if t is None or t.type != TType.RBracket:
                self.report_parse_err(tok, "Ожидалась ]")
            return id_list

    def parse_conn_stmt(self):
        ident_main = self.consume()
        if ident_main is None or ident_main.type != TType.Id:
            self.report_parse_err(ident_main, "Ожидался идентификатор вершины")

        id_main = 0
        try:
            id_main = self.vertices_dict[ident_main.value]
        except KeyError:
            self.report_parse_err(ident_main, "Неизвестная вершина")
        
        arrow = self.consume()
        if arrow is None or arrow.type != TType.Arrow:
            self.report_parse_err(ident_main, "Ожидалось >")

        conns = self.parse_ident_conn()

        weight = self.consume()
        if weight is None or weight.type != TType.Numlit:
            self.report_parse_err(weight, "Ожидался вес соединения между вершинами в виде числа")

        for v in conns:
            self.vertex_connections[(id_main, v)] = int(weight.value)

        tok = self.consume()
        if tok is None or tok.type != TType.Semicolon:
            self.report_parse_err(tok, "Ожидалась точка с запятой")
        
    def parse_conns(self):
        t = self.consume()
        if t is None or t.type != TType.LBrace:
            self.report_parse_err(t, "Ожидалась {{")

        self.parse_graph_kind()
        
        while (t := self.peek()) is not None and (t.type != TType.RBrace and t.type != TType.Eof):
            self.parse_conn_stmt()

        t = self.consume()
        if t is None or t.type != TType.RBrace:
            self.report_parse_err(t, "Ожидалась }}")

    def parse_command(self):
        # <команда> [аргументы]
        # аргументы: список чисел и/или идентификаторов или ничего
        ident = self.consume()
        if ident is None or ident.type != TType.Id:
            self.report_parse_err(ident, "Ожидалось название алгоритма")

        if ident.value not in self.ALGS:
            self.report_parse_err(ident, f"Неизвестный алгоритм {ident.value}")
        
        args = []
        while (t := self.peek()) is not None and t.type != TType.Semicolon:
            args.append(self.consume())
        
        tok = self.consume()
        if tok is None or tok.type != TType.Semicolon:
            self.report_parse_err(tok, "Ожидалась точка с запятой")

        self.commands.append(Command(ident, args))
        self.last_command_line = ident.line
        
    def parse_algs(self):
        t = self.consume()
        if t is None or t.type != TType.LBrace:
            self.report_parse_err(t, "Ожидалась {{")
        
        while (t := self.peek()) is not None and (t.type != TType.RBrace and t.type != TType.Eof):
            self.parse_command()

        t = self.consume()
        if t is None or t.type != TType.RBrace:
            self.report_parse_err(t, "Ожидалась }}")

    def parse_property(self):
        prop = self.consume()
        if prop is None or prop.type != TType.Id:
            self.report_parse_err(prop, "Ожидалось название свойства")
        
        if prop.value not in self.PROPS:
            self.report_parse_err(prop, "Неизвестное название свойства")

        value_tok = self.consume()
        match prop.value:
            case p if p in ["vertexsize", "edgewidth"]:
                if value_tok.type != TType.Numlit:
                    self.report_parse_err(value_tok, f"Для свойства {p} ожидалось численное значение")
                else:
                    value = int(value_tok.value)
            case p if p in ["layout", "palette"]:
                if value_tok.type != TType.Id:
                    self.report_parse_err(value_tok, f"Для свойства {p} ожидалось значение в виде идентификатора")
                else:
                    value = value_tok.value

        match prop.value:
            case "layout":
                if value_tok.value not in ["star", "circle", "grid", "random", "kamadakawai", "davidsonharel"]:
                    self.report_parse_err(value_tok, "Неизвестный алгоритм укладки графа")
            case "palette":
                if value_tok.value not in ["heat", "random", "grey"]:
                    self.report_parse_err(value_tok, "Неизвестное название палитры")

        tok = self.consume()
        if tok is None or tok.type != TType.Semicolon:
            self.report_parse_err(tok, "Ожидалась точка с запятой")
        
        self.visual[prop.value] = value

    def parse_visual(self):
        t = self.consume()
        if t is None or t.type != TType.LBrace:
            self.report_parse_err(t, "Ожидалась {{")
        
        while (t := self.peek()) is not None and (t.type != TType.RBrace and t.type != TType.Eof):
            self.parse_property()

        t = self.consume()
        if t is None or t.type != TType.RBrace:
            self.report_parse_err(t, "Ожидалась }}")

    def parse(self):
        if len(self.tokens) == 0:
            return
        
        t = self.consume()
        if t is None or t.value != "vertex":
            self.report_parse_err(t, "Ожидалась секция определения вершин графа")
        
        self.parse_vertices()

        t = self.consume()
        if t is None or t.value != "graph":
            self.report_parse_err(t, "Ожидалась секция определения соединений графа")
        
        self.parse_conns()

        t = self.peek()
        if t is not None and t.type != TType.Eof:
            if t.value == "algs":
                self.consume()
                self.parse_algs()
            elif t.value == "visual":
                self.consume()
                self.parse_visual()
            else:
                self.report_parse_err(t, f"Неожиданный идентификатор {t.value}, ожидалась секция определения команд или визуальных свойств")
        
        t = self.peek()
        if t is not None and t.type != TType.Eof:
            if t.value == "algs":
                self.consume()
                self.parse_algs()
            else:
                self.report_parse_err(t, f"Неожиданный идентификатор {t.value}, ожидалась секция определения команд")

def compile(content: str) -> Tuple[Graph, List[Command], int, Dict[Token, Token]]:
    lexer = Lexer(content)
    lexer.lex()
    parser = Parser(lexer.tokens)
    parser.parse()
    if parser.last_command_line > 0:
        lcl = parser.last_command_line
    else:
        lcl = lexer.tokens[-1].line
    # g, commands = Graph(parser.vertices, parser.vertex_connections, parser.graph_kind), parser.commands
    return Graph(parser.vertices, parser.vertex_connections, parser.graph_kind), parser.commands, parser.last_command_line, parser.visual

def report_alg_err(token: Token | None, msg: str) -> NoReturn:
    if token is None:
        raise ValueError(f"{msg}")
    else:
        raise ValueError(f"{token.line}:{token.col} {msg}")

def exec_alg(g: Graph, com: Command) -> Tuple[Any]:
    args = com.args
    result = []
    match com.func_name.value:
        case "dijkstra":
            s_v = args[0]
            t_v = args[1]
            if s_v.type != TType.Id:
                report_alg_err(s_v, "Ожидался идентификатор вершины")
            if t_v.type != TType.Id:
                report_alg_err(t_v, "Ожидался идентификатор вершины")
            s = next((v.id for v in g.vertices if v.name == s_v.value), -1)
            t = next((v.id for v in g.vertices if v.name == t_v.value), -1)
            if s == -1:
                report_alg_err(s_v, "Вершина не определена")
            if t == -1:
                report_alg_err(t_v, "Вершина не определена")
            r_m = algs.reachability_matrix(g)
            if r_m[s, t] == 0:
                report_alg_err(com.func_name, f"Между вершинами {s_v.value} и {t_v.value} не может быть найден путь")
            d, path = algs.dijkstra(g, s, t)
            result += [d, path]
        case "floyd":
            res = algs.floyd(g)
            if res == False:
                result += [res]
            else:
                result += [res[0], res[1]]
        case "eulerness":
            result += [algs.is_euler(g)]
        case "fleury":
            if not algs.is_euler(g):
                report_alg_err(com.func_name, "Граф не эйлеровый")
            cycle = algs.fleury(g)
            result += [cycle]
        case "degrees":
            degrees = algs.degrees(g)
            result += [degrees]
        case "coloring":
            is_connected = len(algs.conn_comps(g)) == 1
            if not is_connected:
                report_alg_err(com.func_name, "Граф должен быть связным")
            if g.kind != GraphKind.Undirected:
                report_alg_err(com.func_name, "Граф должен быть неориентированным")
            q, d = algs.coloring(g)
            result += [q, d]
        case "connectivity":
            comps = algs.conn_comps(g)
            n_comps = len(comps)
            result += [comps, n_comps]
        case "strongcomps":
            comps = algs.strong_comps(g)
            n_comps = len(comps)
            result += [comps, n_comps]
        case _:
            raise ValueError("Неизвестное название алгоритма")
    result.insert(0, com.func_name.value)
    return tuple(result)

def from_ig_graph(g: ig.Graph) -> str:
    kind = GraphKind.Directed if g.is_directed() else GraphKind.Undirected
    if "v_id" in g.vs.attributes():
        verts = g.vs["v_id"]
    else:
        verts = g.vs["id"]
    n = len(verts)
    if "v_label" in g.vs.attributes():
        labels = g.vs["v_label"]
    else:
        labels = ["" for _ in range(n)]
    if "weight" in g.es.attributes():
        weights = g.es["weight"]
    else:
        weights = [1 for _ in range(len(g.es))]
    es = [(e.source, e.target) for e in g.es]
    vertex_str = [f"{v};" if l == "" else f"{v}: \"{l}\";" for v, l in zip(verts, labels)]
    graph_str = [f"{verts[s]} > {verts[t]} #{int(w)};" for (s, t), w in zip(es, weights)]
    program = f"""
vertex {{
{"\n".join(vertex_str)}
}}

graph {{
{"directed" if kind == GraphKind.Directed else "undirected"};
{"\n".join(graph_str)}
}}
"""
    return program