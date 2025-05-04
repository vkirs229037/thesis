from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import artist
import igraph as ig
from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QPlainTextEdit, QMessageBox, QFileDialog, QComboBox, QLabel
from PyQt6.QtGui import QAction, QKeySequence   
from compiler import compile, ParseError, LexError, exec_alg
from graph import Graph
import algs
import os

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setWindowTitle("Grapher - Новый граф")

        self.graph = None
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.button = QPushButton("Построить")
        self.editor = QPlainTextEdit()
        self.dirty = False
        self.fileName: str | None = None
        self.editor.textChanged.connect(self.change_dirty)
        self.button.clicked.connect(self.plot)

        self.test_btn = QPushButton("Тест")
        self.test_btn.clicked.connect(self.test)

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
        dijkstra_action.triggered.connect(lambda checked, arg="dijkstra": self.alg(checked, arg))
        floyd_action = QAction("Алгоритм Флойда", self)
        floyd_action.triggered.connect(lambda checked, arg="floyd": self.alg(checked, arg))
        chinese_post_action = QAction("Задача китайского почтальона", self)
        chinese_post_action.triggered.connect(lambda checked, arg="chinesepostman": self.alg(checked, arg))
        max_match_action = QAction("Максимальные паросочетания", self)
        max_match_action.triggered.connect(lambda checked, arg="maxmatching": self.alg(checked, arg))
        
        algMenu.addActions([dijkstra_action, floyd_action, chinese_post_action, max_match_action])

        workspace = QWidget()
        workspace_layout = QHBoxLayout()
        graph_editor = QWidget()
        graph_editor_layout = QVBoxLayout()
        graph_editor_layout.addWidget(self.editor)
        graph_editor_layout.addWidget(self.button)
        graph_editor_layout.addWidget(self.test_btn)
        graph_editor.setLayout(graph_editor_layout)
        graph_image = QWidget()
        graph_image_layout = QVBoxLayout()
        graph_image_layout.addWidget(self.toolbar)
        graph_image_layout.addWidget(self.canvas)
        graph_image.setLayout(graph_image_layout)
        workspace_layout.addWidget(graph_editor)
        workspace_layout.addWidget(graph_image)
        workspace.setLayout(workspace_layout)

        container = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(workspace)
        container.setLayout(layout)
        self.setCentralWidget(container)

    def change_dirty(self):
        if not self.dirty:
            self.setWindowTitle(self.windowTitle() + " *")
            self.dirty = True

    def clear_canvas(self):
        self.figure.clear()
        self.canvas.draw()

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
        self.clear_canvas()

    def new_graph(self):
        if not self.confirm_save():
            return
        self.fileName = None
        self.editor.setPlainText("")
        self.dirty = False
        self.setWindowTitle("Grapher - Новый граф")
        self.clear_canvas()

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
        
    def compile(self) -> Graph:
        pass

    def plot(self):
        content = self.editor.toPlainText()
        try:
            graph, commands = compile(content)
        except (LexError, ParseError) as e:
            msgbox = QMessageBox(QMessageBox.Icon.Critical, "Ошибка", f"{e}", QMessageBox.StandardButton.Ok)
            msgbox.exec()
            return
        alg_results = [exec_alg(graph, c) for c in commands]
        print(alg_results)
        ig_graph = graph.to_ig_graph()
        layout = ig_graph.layout("kamada_kawai")
        graph.print()
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        vertex_size = 60
        ig.plot(
            ig_graph,
            target=ax,
            layout=layout,
            vertex_size=vertex_size,
            vertex_label=ig_graph.vs["name"],
            vertex_label_dist=0,
            edge_label=ig_graph.es["weight"],
        )
        for i, dot in enumerate(layout.coords):
            ax.text(dot[0] - vertex_size*0.001, dot[1] - vertex_size*0.0025, ig_graph.vs["label"][i], transform=ax.transData)
        self.canvas.draw()
        self.graph = graph
        print(commands)

    def test(self):
        r_m = algs.reachability_matrix(self.graph)
        print(r_m)

    def alg(self, checked, alg_name):
        self.alg_window = AlgWindow(self.graph, alg_name)
        self.alg_window.show()

    def export_graph(self):
        file_name, file_ext = QFileDialog.getSaveFileName(self, "Экспортировать граф", "", "Изображение PNG(*.png);;Векторное изображение SVG(*.svg);;Все файлы(*)")
        if not file_name:
            return
        if "." in file_ext:
            file_ext = file_ext[file_ext.index("."):-1]
        else:
            file_ext = ""
        self.figure.savefig(file_name + file_ext)


class AlgWindow(QWidget):
    def __init__(self, g: Graph, alg_name: str):
        super().__init__()
        self.setWindowTitle(alg_name)
        self.setBaseSize(QSize(800, 600))
        self.alg_name = alg_name
        self.graph = g
        layout = QVBoxLayout()
        match alg_name:
            case "dijkstra":
                start_label = QLabel("Начальная вершина", self)
                start_vertices = QComboBox()
                start_vertices.addItems(map(str, g.vertices))
                end_label = QLabel("Конечная вершина", self)
                end_vertices = QComboBox()
                end_vertices.addItems(map(str, g.vertices))
                layout.addWidget(start_label)
                layout.addWidget(start_vertices)
                layout.addWidget(end_label)
                layout.addWidget(end_vertices)
            case "floyd":
                pass
            case "chinesepostman":
                pass
            case "maxmatching":
                pass
            case _:
                raise NotImplementedError(f"Окно не реализовано для алгоритма {alg_name}")
        start_button = QPushButton("Добавить алгоритм", self)
        start_button.pressed.connect(self.exec)
        layout.addWidget(start_button)
        self.setLayout(layout)

    def exec(self):
        print(f"{self.alg_name}")