from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import igraph as ig
from PyQt6.QtCore import QSize, QAbstractTableModel, Qt
from PyQt6.QtWidgets import QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QDialog, QDialogButtonBox, QPlainTextEdit, QMessageBox, QFileDialog, QComboBox, QLabel, QTabWidget, QTableView, QHeaderView
from PyQt6.QtGui import QAction, QKeySequence, QTextCursor
from compiler import compile, ParseError, LexError, exec_alg
from graph import Graph
import os
from typing import Tuple, Any
import numpy as np

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
        export_action = QAction("Экспорт изображения", self)
        export_action.triggered.connect(self.export_graph)
        export_action.setShortcut(QKeySequence("Ctrl+e"))
        exit_action = QAction("Выход", self)
        exit_action.triggered.connect(self.close)
        exit_action.setShortcut(QKeySequence("Ctrl+q"))
        fileMenu = menu.addMenu("Файл")
        fileMenu.addActions([new_action, load_action, save_action])
        fileMenu.addSeparator()
        fileMenu.addAction(export_action)
        fileMenu.addSeparator()
        fileMenu.addAction(exit_action)

        actionMenu = menu.addMenu("Действия с графом")
        algMenu = actionMenu.addMenu("Алгоритмы")

        dijkstra_action = QAction("Алгоритм Дейкстры", self)
        dijkstra_action.triggered.connect(lambda checked, arg="dijkstra": self.insert_alg(checked, arg))
        floyd_action = QAction("Алгоритм Флойда", self)
        floyd_action.triggered.connect(lambda checked, arg="floyd": self.insert_alg(checked, arg))
        fleury_action = QAction("Нахождение эйлерового цикла", self)
        fleury_action.triggered.connect(lambda checked, arg="fleury": self.insert_alg(checked, arg))
        chinese_post_action = QAction("Задача китайского почтальона", self)
        chinese_post_action.triggered.connect(lambda checked, arg="chinesepostman": self.insert_alg(checked, arg))
        max_match_action = QAction("Максимальные паросочетания", self)
        max_match_action.triggered.connect(lambda checked, arg="maxmatching": self.insert_alg(checked, arg))
        max_indep_action = QAction("Максимальное независимое множество", self)
        max_indep_action.triggered.connect(lambda checked, arg="maxindepset": self.insert_alg(checked, arg))
        abs_center_action = QAction("Абсолютный p-центр", self)
        abs_center_action.triggered.connect(lambda checked, arg="abscenter": self.insert_alg(checked, arg))
        neg_cycle_action = QAction("Нахождение циклов отрицательного веса", self)
        neg_cycle_action.triggered.connect(lambda checked, arg="negcycle": self.insert_alg(checked, arg))
        all_trees_action = QAction("Все деревья графа", self)
        all_trees_action.triggered.connect(lambda checked, arg="alltrees": self.insert_alg(checked, arg))
        algMenu.addActions([dijkstra_action, floyd_action, fleury_action, chinese_post_action, max_match_action, max_indep_action, abs_center_action, neg_cycle_action, all_trees_action])

        propertyMenu = actionMenu.addMenu("Свойства графа")

        degrees_action = QAction("Степени вершин", self)
        degrees_action.triggered.connect(lambda checked, arg="degrees": self.insert_alg(checked, arg))
        connected_action = QAction("Связность графа", self)
        connected_action.triggered.connect(lambda checked, arg="connectivity": self.insert_alg(checked, arg))
        planarity_action = QAction("Планарность графа", self)
        planarity_action.triggered.connect(lambda checked, arg="planarity": self.insert_alg(checked, arg))
        euler_action = QAction("Эйлеровость графа", self)
        euler_action.triggered.connect(lambda checked, arg="eulerness": self.insert_alg(checked, arg))
        chrom_num_action = QAction("Хроматическое число графа", self)
        chrom_num_action.triggered.connect(lambda checked, arg="chromnum": self.insert_alg(checked, arg))
        propertyMenu.addActions([degrees_action, connected_action, planarity_action, euler_action, chrom_num_action])

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
        self.graph_images.clear()
        self.cur_img = -1

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
            graph, commands, self.last_command_line = compile(content)
            alg_results = [exec_alg(graph, c) for c in commands]
        except (LexError, ParseError, ValueError) as e:
            msgbox = QMessageBox(QMessageBox.Icon.Critical, "Ошибка", f"{e}", QMessageBox.StandardButton.Ok)
            msgbox.exec()
            return
        print(alg_results)
        ig_graph = graph.to_ig_graph()
        graph.print()
        if len(alg_results) == 0:
            self.plot_one(ig_graph, None)
        else:
            self.results = alg_results
            for r in alg_results:
                self.plot_one(ig_graph, r)
        self.graph = graph

    def draw_ax_text(self, ax: Axes, text: str, pos_x: float, pos_y: float, color: str = "k"):
        ax.text(pos_x, pos_y, text, ha="center", va="baseline", c=color)

    def draw_figure_text(self, text: str):
        self.figures[self.cur_img].text(0.05, 0.05, text)

    def plot_one(self, g: ig.Graph, result: Tuple[Any] | None):
        print(result)
        f = Figure()
        f.subplots_adjust(left=0, bottom=0, right=1, top=1)
        self.figures.append(f)
        self.cur_img += 1
        ax = self.figures[self.cur_img].add_subplot()
        layout = g.layout("kamada_kawai")
        vertex_size = 60
        g.es["color"] = "black"
        g.vs["color"] = "white"
        if result is not None:
            title = result[0]
            match result[0]:
                case "dijkstra":
                    self.draw_figure_text(f"Длина кратчайшего пути: {result[1]}")
                    path_vs = g.vs.select(result[2])
                    path_es = g.es.select(_source_in = path_vs[:-1], _target_in = path_vs[1:])
                    path_vs["color"] = "red"
                    path_es["color"] = "red"
                case "floyd":
                    self.draw_figure_text("Результат отобразится по нажатию кнопки \"Показать результат\"")
                case "fleury":
                    cycle = ", ".join([f"{g.vs["name"][t[0]]} - {g.vs["name"][t[1]]}" for t in result[1]])
                    self.draw_figure_text(f"Цикл: {cycle}")
                    cycle_es = g.es.select(_source_in = [t[0] for t in result[1]], _target_in = [t[1] for t in result[1]])
                    cycle_es["color"] = "red"
                case "degrees":
                    for i, dot in enumerate(layout.coords):
                        self.draw_ax_text(ax, result[1][i], dot[0], dot[1] + vertex_size*0.003, "r")
                case _:
                    raise ValueError
        else:
            title = "Граф"
        ig.plot(
            g,
            target=ax,
            layout=layout,
            vertex_size=vertex_size,
            vertex_label=g.vs["name"],
            vertex_label_dist=0,
            edge_label=g.es["weight"],
            vertex_color=g.vs["color"],
            edge_color=g.es["color"]
        )
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

    def show_current_result(self):
        raw_result = self.results[self.cur_img]
        title = raw_result[0]
        result = []
        match raw_result[0]:
            case "floyd":
                v_names = np.array(list(map(lambda v: v.name, self.graph.vertices)))
                headers = list(map(lambda v: str(v), v_names))
                result.append(("table", "Таблица расстояний", raw_result[1][0]))
                result.append(("table", "Таблица путей", v_names[raw_result[1][1]]))
            case "fleury":
                headers = None
                cycle = ", ".join([f"{self.graph.vertices[t[0]].name} - {self.graph.vertices[t[1]].name}" for t in raw_result[1]])
                print(cycle)
                result.append(("text", "Эйлеров цикл", cycle))
        self.res_window = ResultWindow(title, result, headers)
        self.res_window.show()

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

    def export_graph(self):
        file_name, file_ext = QFileDialog.getSaveFileName(self, "Экспортировать граф", "", "Изображение PNG(*.png);;Векторное изображение SVG(*.svg);;Все файлы(*)")
        if not file_name:
            return
        if "." in file_ext:
            file_ext = file_ext[file_ext.index("."):-1]
        else:
            file_ext = ""
        self.figure.savefig(file_name + file_ext)


class AlgWindow(QDialog):
    def __init__(self, g: Graph, alg_name: str):
        super().__init__()
        self.setWindowTitle(alg_name)
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
            case "chinesepostman":
                pass
            case "maxmatching":
                pass
            case "fleury":
                pass
            case "degrees":
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
        self.setWindowTitle(title)
        layout = QVBoxLayout()
        self.tabs = QTabWidget()
        for dt, header, data in result:
            print(data)
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