from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import igraph as ig
from PyQt6.QtCore import QSize, QAbstractTableModel, Qt
from PyQt6.QtWidgets import QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QDialog, QDialogButtonBox, QPlainTextEdit, QMessageBox, QFileDialog, QComboBox, QLabel, QTabWidget, QTableView, QHeaderView
from PyQt6.QtGui import QAction, QKeySequence, QTextCursor
from compiler import compile, ParseError, LexError, exec_alg, from_ig_graph
from graph import Graph
import os
from typing import Tuple, Any
import numpy as np
from pandas import DataFrame
import utils

ALG_NAME_TABLE = {
    "dijkstra": "Кратчайший путь",
    "fleury": "Эйлеровый цикл",
    "floyd": "Все кратчайшие пути",
    "degrees": "Степени всех вершин",
    "eulerness": "Эйлеровость графа",
    "connectivity": "Связность графа",
    "coloring": "Раскраска графа"
}

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setWindowTitle("Grapher - Новый граф")
        self.resize(800, 600)

        self.graph = None
        self.last_command_line = -1
        self.cur_img = -1
        self.figures = [Figure()]
        self.figures[0].subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.canvases = [FigureCanvas(self.figures[0])]
        self.toolbars = [NavigationToolbar(self.canvases[0], self)]
        self.button = QPushButton("Построить")
        self.editor = QPlainTextEdit()
        self.dirty = False
        self.fileName: str | None = None
        self.editor.textChanged.connect(self.change_dirty)
        self.button.clicked.connect(self.plot)
        self.results = []
        self.ig_graphs = []
        self.visual = {}

        self.alg_name = ""

        menu = self.menuBar()
        
        new_action = QAction("Новый граф", self)
        new_action.triggered.connect(self.new_graph)
        new_action.setShortcut(QKeySequence("Ctrl+n"))
        load_action = QAction("Загрузить граф", self)
        load_action.triggered.connect(self.load_graph)
        load_action.setShortcut(QKeySequence("Ctrl+o"))
        save_action = QAction("Сохранить граф", self)
        save_action.triggered.connect(self.save_graph)
        save_action.setShortcut(QKeySequence("Ctrl+s"))
        self.export_action = QAction("Экспорт изображения", self)
        self.export_action.triggered.connect(self.export_image)
        self.export_action.setShortcut(QKeySequence("Ctrl+e"))
        self.save_result_action = QAction("Сохранить результаты анализа", self)
        self.save_result_action.triggered.connect(self.save_results)
        self.save_result_action.setShortcut(QKeySequence("Ctrl+r"))
        exit_action = QAction("Выход", self)
        exit_action.triggered.connect(self.close)
        exit_action.setShortcut(QKeySequence("Ctrl+q"))
        fileMenu = menu.addMenu("Файл")
        fileMenu.addActions([new_action, load_action, save_action])
        fileMenu.addSeparator()
        fileMenu.addActions([self.export_action, self.save_result_action])
        fileMenu.addSeparator()
        fileMenu.addAction(exit_action)

        self.actionMenu = menu.addMenu("Действия с графом")
        algMenu = self.actionMenu.addMenu("Алгоритмы")

        dijkstra_action = QAction("Алгоритм Дейкстры", self)
        dijkstra_action.triggered.connect(lambda checked, arg="dijkstra": self.insert_alg(checked, arg))
        floyd_action = QAction("Алгоритм Флойда", self)
        floyd_action.triggered.connect(lambda checked, arg="floyd": self.insert_alg(checked, arg))
        fleury_action = QAction("Нахождение эйлерового цикла", self)
        fleury_action.triggered.connect(lambda checked, arg="fleury": self.insert_alg(checked, arg))
        coloring_action = QAction("Раскраска графа", self)
        coloring_action.triggered.connect(lambda checked, arg="coloring": self.insert_alg(checked, arg))
        algMenu.addActions([dijkstra_action, floyd_action, fleury_action, coloring_action])

        propertyMenu = self.actionMenu.addMenu("Свойства графа")

        degrees_action = QAction("Степени вершин", self)
        degrees_action.triggered.connect(lambda checked, arg="degrees": self.insert_alg(checked, arg))
        connected_action = QAction("Связность графа", self)
        connected_action.triggered.connect(lambda checked, arg="connectivity": self.insert_alg(checked, arg))
        planarity_action = QAction("Планарность графа", self)
        planarity_action.triggered.connect(lambda checked, arg="planarity": self.insert_alg(checked, arg))
        euler_action = QAction("Эйлеровость графа", self)
        euler_action.triggered.connect(lambda checked, arg="eulerness": self.insert_alg(checked, arg))
        propertyMenu.addActions([degrees_action, connected_action, planarity_action, euler_action])

        self.ie_menu = menu.addMenu("Импорт и экспорт")

        import_menu = self.ie_menu.addMenu("Импорт")

        from_graphml = QAction("GraphML...", self)
        from_graphml.triggered.connect(self.import_from)
        import_menu.addAction(from_graphml)

        export_menu = self.ie_menu.addMenu("Экспорт")
        to_graphml = QAction("GraphML...", self)
        to_graphml.triggered.connect(lambda checked, arg="graphml": self.export_to(checked, arg))
        to_dot = QAction("DOT...", self)
        to_dot.triggered.connect(lambda checked, arg="dot": self.export_to(checked, arg))
        export_menu.addActions([to_graphml, to_dot])

        self.help_menu = menu.addMenu("Справка")

        about_action = QAction("О программе", self)
        about_action.triggered.connect(self.about)
        self.help_menu.addAction(about_action)

        workspace = QWidget()
        workspace_layout = QHBoxLayout()
        graph_editor = QWidget()
        graph_editor_layout = QVBoxLayout()
        graph_editor_layout.addWidget(self.editor)
        graph_editor_layout.addWidget(self.button)
        graph_editor.setLayout(graph_editor_layout)

        self.graph_images = QTabWidget()
        self.graph_images.currentChanged.connect(self.on_tab_change)
        
        workspace_layout.addWidget(graph_editor)
        workspace_layout.addWidget(self.graph_images)
        workspace.setLayout(workspace_layout)

        container = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(workspace)
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.toggle_menu(False)

    def about(self):
        aboutbox = AboutBox()
        aboutbox.exec()

    def toggle_menu(self, value: bool):
        self.export_action.setEnabled(value)
        self.save_result_action.setEnabled(value)
    
    def on_tab_change(self, i):
        self.cur_img = i

    def change_dirty(self):
        if not self.dirty:
            self.setWindowTitle(self.windowTitle() + " *")
            self.dirty = True

    def clear_images(self):
        self.figures = []
        self.canvases = []
        self.toolbars = []
        self.results = []
        self.ig_graphs = []
        self.graph_images.clear()
        self.cur_img = -1
        self.toggle_menu(False)

    def save_graph(self):
        code = self.editor.toPlainText()
        if self.fileName is None:
            self.fileName, _ = QFileDialog.getSaveFileName(self, "Сохранить граф", "", "Файл графа Grapher(*.gph);;Все файлы(*)")
            if not self.fileName:
                return
        self.setWindowTitle("Grapher - " + os.path.basename(self.fileName))
        self.fileName = self.fileName.removesuffix(".gph")
        with open(self.fileName + ".gph", "w") as f:
            f.write(code)
        self.dirty = False

    def load_graph(self):
        if not self.confirm_save():
            return
        file_name, _ = QFileDialog.getOpenFileName(self, "Загрузить граф", "", "Файл графа Grapher(*.gph);;Все файлы(*)")
        if not file_name:
            return
        self.fileName = file_name
        with open(self.fileName, "r") as f:
            code = f.read()
            self.editor.setPlainText(code)
        self.setWindowTitle("Grapher - " + os.path.basename(self.fileName))
        self.dirty = False
        self.clear_images()

    def new_graph(self):
        if not self.confirm_save():
            return
        self.fileName = None
        self.editor.setPlainText("")
        self.dirty = False
        self.setWindowTitle("Grapher - Новый граф")
        self.clear_images()

    def closeEvent(self, e):
        if self.confirm_save():
            e.accept()
        else:
            e.ignore()

    def confirm_save(self):
        if self.dirty:
            msgbox = QMessageBox(QMessageBox.Icon.Question, "", "Сохранить текущий граф?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel)
            res = msgbox.exec()
            match res:
                case QMessageBox.StandardButton.Yes:
                    self.save_graph()
                    return True
                case QMessageBox.StandardButton.No:
                    self.dirty = False
                    return True
                case QMessageBox.StandardButton.Cancel:
                    return False
        else:
            return True

    def plot(self):
        self.clear_images()
        content = self.editor.toPlainText()
        try:
            graph, commands, self.last_command_line, self.visual = compile(content)
        except (LexError, ParseError) as e:
            msgbox = QMessageBox(QMessageBox.Icon.Critical, "Ошибка", f"{e}", QMessageBox.StandardButton.Ok)
            msgbox.exec()
            return
        self.graph = graph
        alg_results = []
        for c in commands:
            try:
                alg_results.append(exec_alg(graph, c))
            except ValueError as e:
                msgbox = QMessageBox(QMessageBox.Icon.Critical, "Ошибка", f"{e}", QMessageBox.StandardButton.Ok)
                msgbox.exec()
            except NotImplementedError:
                msgbox = QMessageBox(QMessageBox.Icon.Critical, "Ошибка", f"Алгоритм не реализован!", QMessageBox.StandardButton.Ok)
                msgbox.exec()
        ig_graph = graph.to_ig_graph()
        if len(alg_results) == 0:
            self.plot_one(ig_graph, None)
        else:
            self.results = alg_results
            for r in alg_results:
                self.plot_one(ig_graph, r)
        self.cur_img = 0
        self.toggle_menu(True)

    def draw_ax_text(self, ax: Axes, text: str, pos_x: float, pos_y: float, color: str = "k"):
        ax.text(pos_x, pos_y, text, ha="center", va="baseline", c=color)

    def draw_figure_text(self, text: str):
        self.figures[self.cur_img].text(0.05, 0.05, text)

    def plot_one(self, g: ig.Graph, result: Tuple[Any] | None):
        f = Figure()
        f.subplots_adjust(left=0, bottom=0, right=1, top=1)
        self.figures.append(f)
        self.cur_img += 1
        ax = self.figures[self.cur_img].add_subplot()
        if (layout_id := self.visual.get("layout")) is None:
            layout_id = "circle"
        if layout_id == "kamadakawai": layout_id = "kamada_kawai"
        if layout_id == "davidsonharel": layout_id = "davidson_harel"
        if layout_id == "kamada_kawai":
            print(g.es["weight"])
            layout = g.layout(layout_id, weights=g.es["weight"])
        else:
            layout = g.layout(layout_id)
        if (edge_width := self.visual.get("edgewidth")) is None:
            match self.graph.n:
                case n if n < 10:
                    edge_width = 2
                case n if n < 20:
                    edge_width = 1
                case _:
                    edge_width = 0.5
        if (vertex_size := self.visual.get("vertexsize")) is None:
            match self.graph.n:
                case n if n < 10:
                    vertex_size = 60
                case n if n < 20:
                    vertex_size = 30
                case _:
                    vertex_size = 10
        g.vs["color"] = [v.id for v in self.graph.vertices]
        g.es["color"] = "black"
        palette_n = self.graph.n
        if result is not None:
            title = ALG_NAME_TABLE[result[0]]
            match result[0]:
                case "dijkstra":
                    self.draw_figure_text(f"Длина кратчайшего пути: {result[1]}")
                    path_vs = g.vs.select(result[2])
                    for i in range(len(path_vs) - 1):
                        e = g.get_eid(path_vs[i], path_vs[i + 1])
                        g.es[e]["color"] = "red"
                    path_vs["color"] = "red"
                case "floyd":
                    self.draw_figure_text("Результат отобразится по нажатию кнопки \"Показать результат\"")
                case "fleury":
                    cycle = ", ".join([f"{g.vs["id"][t[0]]} - {g.vs["id"][t[1]]}" for t in result[1]])
                    self.draw_figure_text(f"Цикл: {cycle}")
                    cycle_es = g.es.select(_source_in = [t[0] for t in result[1]], _target_in = [t[1] for t in result[1]])
                    cycle_es["color"] = "red"
                case "degrees":
                    for i, dot in enumerate(layout.coords):
                        self.draw_ax_text(ax, result[1][i], dot[0], dot[1] + vertex_size*0.003, "r")
                case "eulerness":
                    if result[1] == True:
                        answer = "Граф эйлеровый"
                    else:
                        answer = "Граф не эйлеровый"
                    self.draw_figure_text(answer)
                case "connectivity":
                    answer = "Граф "
                    if result[2] == 1:
                        answer += "связный"
                    else:
                        answer += "несвязный"
                    answer += f", число компонент связности {result[2]}"
                    palette_n = result[2]
                    self.draw_figure_text(answer)
                    c = 0
                    for comp in result[1]:
                        comp_vs = g.vs.select(comp)
                        comp_vs["color"] = c
                        c += 1
                case "coloring":
                    self.draw_figure_text(f"Хроматическое число графа - {result[1]}")
                    palette_n = result[1]
                    for v in result[2]:
                        c_vs = g.vs.select(v)
                        c_vs["color"] = result[2][v]
                case _:
                    raise ValueError
        else:
            title = "Граф"
        if (pal_id := self.visual.get("palette")) is None:
            pal_id = "grey"
        pal = ig.PrecalculatedPalette(utils.gen_palette(pal_id, palette_n))
        ig.plot(
            g,
            target=ax,
            layout=layout,
            vertex_size=vertex_size,
            edge_width=edge_width,
            vertex_label=g.vs["id"],
            vertex_label_dist=0,
            edge_label=g.es["weight"],
            vertex_color=g.vs["color"],
            edge_color=g.es["color"],
            palette = pal
        )
        self.ig_graphs.append(g)
        for i, dot in enumerate(layout.coords):
            self.draw_ax_text(ax, g.vs["label"][i], dot[0], dot[1] - vertex_size*0.005)
        canvas = FigureCanvas(self.figures[self.cur_img])
        self.canvases.append(canvas)
        toolbar = NavigationToolbar(self.canvases[self.cur_img])
        self.toolbars.append(toolbar)
        open_result_btn = QPushButton("Показать результат")
        open_result_btn.pressed.connect(self.show_current_result)
        graph_tab = QWidget()
        graph_tab_layout = QVBoxLayout()
        graph_tab_layout.addWidget(self.toolbars[self.cur_img])
        graph_tab_layout.addWidget(self.canvases[self.cur_img])
        graph_tab_layout.addWidget(open_result_btn)
        graph_tab.setLayout(graph_tab_layout)
        self.graph_images.addTab(graph_tab, title)

    def human_readable_result(self, alg_name: str, raw: list[Any]) -> list[Tuple[str | np.ndarray]]:
        result = []
        match alg_name:
            case "dijkstra":
                dist = raw[0]
                path = " - ".join([f"{self.graph.vertices[i].name}" for i in raw[1]])
                result.append(("text", "Кратчайший путь", f"Расстояние: {dist}\nПуть: {path}"))
            case "degrees":
                degs = "\n".join([f"{self.graph.vertices[i].name}: {raw[0][i]}" for i in range(self.graph.n)])
                result.append(("text", "Степени вершин", degs))
            case "eulerness":
                if raw[0]:
                    answer = "Граф эйлеров"
                else:
                    answer = "Граф не эйлеров"
                result.append(("text", "Эйлеровость графа", answer))
            case "floyd":
                v_names = [str(v) for v in self.graph.vertices]
                result.append(("table", "Таблица расстояний", raw[0]))
                result.append(("table", "Таблица путей", v_names[raw[1]]))
            case "fleury":
                cycle = ", ".join([f"{self.graph.vertices[t[0]].name} - {self.graph.vertices[t[1]].name}" for t in raw[0]])
                result.append(("text", "Эйлеров цикл", cycle))
            case "connectivity":
                answer = "Граф "
                if raw[1] == 1:
                    answer += "связный"
                else:
                    answer += "несвязный"
                answer += f", число компонент связности {raw[1]}"
                result.append(("text", "Связность", answer))
                v_names = np.array([str(v) for v in self.graph.vertices])
                comps = [v_names[list(c)] for c in raw[0]]
                comps_str_arr = []
                for i, comp in enumerate(comps):
                    comps_str_arr.append(f"Компонента {i + 1}: {", ".join(list(comp))}")
                comps_str = "\n".join(comps_str_arr)
                result.append(("text", "Компоненты связности", comps_str))
            case "coloring":
                result.append(("text", "Хроматическое число графа", str(raw[0])))
                answer = ""
                color_to_v = {i: [] for i in raw[1].values()}
                for v in self.graph.vertices:
                    color_to_v[raw[1][v.id]].append(v.id)
                for col in color_to_v:
                    color_str = f"Цвет {col+1}: "
                    for v in color_to_v[col]:
                        color_str += f"{self.graph.vertices[v]} "
                    answer += color_str + "\n"
                result.append(("text", "Раскраска графа", answer))
            case _:
                raise NotImplementedError
        return result

    def show_current_result(self):
        if len(self.results) == 0:
            msgbox = QMessageBox(QMessageBox.Icon.Warning, "Ошибка", "Нет результатов для отображения", QMessageBox.StandardButton.Ok)
            msgbox.exec()
            return
        raw_result = self.results[self.cur_img]
        title = raw_result[0]
        result = []
        headers = list(map(lambda v: v.name, self.graph.vertices))
        result = self.human_readable_result(raw_result[0], raw_result[1:])
        self.res_window = ResultWindow(title, result, headers)
        self.res_window.show()

    def save_results(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Сохранить результаты анализа", "", "Текстовый файл(*.txt);;Все файлы(*)")
        filename = filename.removesuffix(".txt")
        with open(filename + ".txt", "w") as f:
            v_names = [str(v) for v in self.graph.vertices]
            report = []
            for result in self.results:
                h_r_res = self.human_readable_result(result[0], result[1:])
                report.append(ALG_NAME_TABLE[result[0]])
                for data in h_r_res:
                    match data[0]:
                        case "text":
                            report.append(data[2])
                        case "table":
                            report.append(data[1])
                            df = DataFrame(data[2], columns=v_names, index=v_names)
                            report.append(str(df))
                report.append("\n")
            f.write("\n".join(report))

    def insert_alg(self, checked, alg_name):
        self.alg_window = AlgWindow(self.graph, alg_name)
        self.alg_window.exec()
        command = self.alg_window.command
        if command == "":
            return
        lcl = self.last_command_line
        if "algs" in self.editor.toPlainText():
            offset = 0
            block = self.editor.document().findBlockByLineNumber(lcl)
            if "}" in block.text():
                offset = 1
            cursor = QTextCursor(self.editor.document().findBlockByLineNumber(lcl))
            cursor.movePosition(QTextCursor.MoveOperation.EndOfLine, QTextCursor.MoveMode.MoveAnchor)
            cursor.movePosition(QTextCursor.MoveOperation.Left, QTextCursor.MoveMode.MoveAnchor, offset)
            self.editor.setTextCursor(cursor)
            self.editor.insertPlainText(f"\n{command}")
            self.last_command_line += 1
        else:
            self.editor.moveCursor(QTextCursor.MoveOperation.End)
            self.editor.insertPlainText(f"\nalgs {{\n {command}\n}}")
            self.last_command_line += 2

    def export_image(self):
        file_name, file_ext = QFileDialog.getSaveFileName(self, "Экспортировать граф", "", "Изображение PNG(*.png);;Векторное изображение SVG(*.svg);;Все файлы(*)")
        if not file_name:
            return
        if "." in file_ext:
            file_ext = file_ext[file_ext.index("."):-1]
        else:
            file_ext = ""
        if len(self.results) == 0:
            self.figures[0].savefig(file_name + file_ext)
        else:
            for fig, res in zip(self.figures, self.results):
                fig.savefig(file_name + "_" + res[0] + file_ext)

    def import_from(self):
        if not self.confirm_save():
            return
        file_name, _ = QFileDialog.getOpenFileName(self, "Загрузить граф GraphML", "", "Файл графа GraphML(*.graphml);;Все файлы(*)")
        if not file_name:
            return
        try:
            ig_graph = ig.Graph.Read_GraphML(file_name)
            code = from_ig_graph(ig_graph)
        except Exception as e:
            msgbox = QMessageBox(QMessageBox.Icon.Critical, "Ошибка", f"{e}", QMessageBox.StandardButton.Ok)
            msgbox.exec()
            return
        self.editor.setPlainText(code)
        self.setWindowTitle("Grapher - Новый граф")
        self.dirty = False
        self.clear_images()

    def export_to(self, checked, name):
        file_format = "." + name
        human_name = "GraphML" if name == "graphml" else "DOT"
        file_name, _ = QFileDialog.getSaveFileName(self, "Экспортировать граф", "", f"Файл графа {human_name}(*{file_format});;Все файлы(*)")
        if not file_name:
            return
        if len(self.results) == 0:
            ig.Graph.write(self.ig_graphs[0], file_name + file_format, format=name)
        else:
            for g, res in zip(self.ig_graphs, self.results):
                ig.Graph.write(g, file_name + "_" + res[0] + file_format, format=name)


class AlgWindow(QDialog):
    def __init__(self, g: Graph, alg_name: str):
        super().__init__()
        self.setWindowTitle(ALG_NAME_TABLE[alg_name])
        self.setBaseSize(QSize(800, 600))
        self.alg_name = alg_name
        self.graph = g
        self.command = ""
        layout = QVBoxLayout()
        match alg_name:
            case "dijkstra":
                start_label = QLabel("Начальная вершина", self)
                self.start_vertices = QComboBox()
                self.start_vertices.addItems(map(str, g.vertices))
                end_label = QLabel("Конечная вершина", self)
                self.end_vertices = QComboBox()
                self.end_vertices.addItems(map(str, g.vertices))
                layout.addWidget(start_label)
                layout.addWidget(self.start_vertices)
                layout.addWidget(end_label)
                layout.addWidget(self.end_vertices)
            case "floyd":
                pass
            case "fleury":
                pass
            case "eulerness":
                pass
            case "degrees":
                pass
            case "connectivity": 
                pass
            case "coloring":
                pass
            case _:
                raise NotImplementedError(f"Окно не реализовано для алгоритма {alg_name}")
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.make_command)
        self.buttons.rejected.connect(self.close)
        layout.addWidget(self.buttons)
        self.setLayout(layout)

    def make_command(self):
        command = []
        match self.alg_name:
            case "dijkstra":
                command.append("dijkstra")
                s = self.start_vertices.currentIndex()
                t = self.end_vertices.currentIndex()
                command.append(self.graph.vertices[s].name)
                command.append(self.graph.vertices[t].name)
                self.command = " ".join(command) + ";"
            case "floyd":
                self.command = "floyd;"
            case "fleury":
                self.command = "fleury;"
            case "degrees":
                self.command = "degrees;"
            case "eulerness":
                self.command = "eulerness;"
            case "connectivity":
                self.command = "connectivity;"
            case "coloring":
                self.command = "coloring;"
            case _:
                raise ValueError
        self.close()


class TableModel(QAbstractTableModel):
    def __init__(self, data, headers = None):
        super().__init__()
        self._data = data
        self.custom_headers = headers

    def data(self, index, role):
        if role == Qt.ItemDataRole.DisplayRole:
            value = self._data[index.row()][index.column()]
            if isinstance(value, np.int64):
                return int(value)
            else:
                return str(value)
        
    def rowCount(self, index):
        return len(self._data)
    
    def columnCount(self, index):
        if self.rowCount(0) == 0:
            return 0
        return len(self._data[0])

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            return self.custom_headers[section]
        return super().headerData(section, orientation, role)

class ResultWindow(QDialog):
    def __init__(self, title: str, result: list[Tuple[str, str, Any]], headers: list[str] | None):
        super().__init__()
        self.setWindowTitle(ALG_NAME_TABLE[title])
        layout = QVBoxLayout()
        self.tabs = QTabWidget()
        for dt, header, data in result:
            tab = QWidget()
            tab_layout = QVBoxLayout()
            match dt:
                case "table":
                    table = QTableView(self)
                    model = TableModel(data, headers)
                    table_header = table.horizontalHeader()
                    table_header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
                    table.setModel(model)
                    tab_layout.addWidget(table)
                case "text":
                    text = QPlainTextEdit(self)
                    text.setPlainText(data)
                    text.setReadOnly(True)
                    tab_layout.addWidget(text)
            tab.setLayout(tab_layout)
            self.tabs.addTab(tab, header)
        layout.addWidget(self.tabs)
        btn = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok, self)
        btn.accepted.connect(self.close)
        layout.addWidget(btn)
        self.setLayout(layout)

class AboutBox(QDialog):    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("О программе")
        self.setBaseSize(300, 200) 
        
        layout = QVBoxLayout()
        
        title_label = QLabel("Программа для визуализации и анализа графов")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        
        version_label = QLabel("Версия 1.0.0")
        
        uni_label = QLabel("Научно-исследовательский университет \"Московский энергетический институт\", институт Информационных и вычислительных технологий, кафедра Прикладной математики и информатики")
        uni_label.setWordWrap(True)
        desc_label = QLabel("Выпускная квалификационная работа на тему \"Методы и средства визуализации и анализа графов\"")
        desc_label.setWordWrap(True)
        author_label = QLabel("Выполнил студент группы А-13-21 Кирсанов В. Р.")
        author_label.setWordWrap(True)
        
        # Кнопка закрытия
        close_button = QPushButton("ОК")
        close_button.clicked.connect(self.close)
        
        # Добавление элементов в лейаут
        layout.addWidget(title_label)
        layout.addWidget(version_label)
        layout.addStretch(1)
        layout.addWidget(uni_label)
        layout.addWidget(desc_label)
        layout.addWidget(author_label)
        layout.addStretch(1)
        layout.addWidget(close_button)
        
        self.setLayout(layout)
