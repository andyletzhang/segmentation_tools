import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget, QFileDialog, QMessageBox)
from PyQt6.Qsci import QsciScintilla, QsciLexerPython
from PyQt6.QtGui import QFont, QColor, QAction
from PyQt6.QtCore import Qt
import ast

class ExecutionInterrupted(Exception):
    """Custom exception to stop script execution."""
    pass
class InterruptInjector(ast.NodeTransformer):
    """AST Transformer that inserts check_interrupt() into loops."""

    def visit_For(self, node):
        """Inject check_interrupt() at the start of for loops."""
        node.body.insert(0, ast.Expr(value=ast.Call(func=ast.Name(id="check_interrupt", ctx=ast.Load()), args=[], keywords=[])))
        return self.generic_visit(node)

    def visit_While(self, node):
        """Inject check_interrupt() at the start of while loops."""
        node.body.insert(0, ast.Expr(value=ast.Call(func=ast.Name(id="check_interrupt", ctx=ast.Load()), args=[], keywords=[])))
        return self.generic_visit(node)

class ScriptWindow(QMainWindow):
    def __init__(self, parent=None, local_env=None, global_env=None):
        super().__init__()
        self.main_window = parent
        self.setWindowTitle("Script Editor")
        self.setGeometry(100, 100, 800, 500)
        self.script_path = None
        self._setup_menu()

        # Execution environment
        self.local_env = local_env if local_env is not None else {}
        self.global_env = global_env if global_env is not None else {}

        # Main widget and layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()

        # Script editor (QsciScintilla)
        self.text_edit = QsciScintilla(self)
        self.setup_editor()
        layout.addWidget(self.text_edit)

        run_interrupt_layout=QHBoxLayout()
        self.execute_button = QPushButton("Run", self)
        self.execute_button.clicked.connect(self.execute_script)
        self.interrupt_button = QPushButton("Interrupt", self)
        self.execute_button.setShortcut("Ctrl+Return")
        self.interrupt_button.clicked.connect(self.interrupt_execution)
        run_interrupt_layout.addWidget(self.execute_button)
        run_interrupt_layout.addWidget(self.interrupt_button)

        layout.addLayout(run_interrupt_layout)

        central_widget.setLayout(layout)

    @property
    def scripts_path(self):
        if hasattr(self,'_scripts_path'):
            return self._scripts_path.as_posix()
        else:
            from platformdirs import user_documents_dir
            from pathlib import Path
            self._scripts_path=Path(user_documents_dir()).joinpath('segmentation_viewer/scripts')
            self._scripts_path.mkdir(parents=True, exist_ok=True)
            return self._scripts_path.as_posix()

    def _setup_menu(self):
        from .qt import create_action
        self.menu_bar=self.menuBar()
        # FILE
        self.file_menu = self.menu_bar.addMenu("File")
        self.file_menu.addAction(create_action("Open Script", self.load_script, self, 'Ctrl+O'))
        self.file_menu.addAction(create_action("Save Script", self.save_script, self, 'Ctrl+S'))
        self.file_menu.addAction(create_action("Save Script As...", self.save_script_as, self, 'Ctrl+Shift+S'))

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            if self.is_executing:
                self.interrupt_execution()
        super().keyPressEvent(event)

    def interrupt_execution(self):
        """Sets the flag to interrupt execution."""
        self.execution_interrupted = True

    def setup_editor(self):
        """Configures QScintilla with Pythonic syntax highlighting and indentation."""
        lexer = QsciLexerPython(self.text_edit)
        self.text_edit.setLexer(lexer)

        # Set a readable font
        font = QFont("Consolas", 12)
        self.text_edit.setFont(font)
        lexer.setFont(font)
        # Background color
        lexer.setPaper(QColor("#2e2e2e"))  # Dark background
        lexer.setColor(QColor("#ffffff"))  # Default text color

        # Syntax Highlighting Colors
        lexer.setColor(QColor("#ffffff"), QsciLexerPython.Default)  # Default text
        lexer.setColor(QColor("#dcdcaa"), QsciLexerPython.Keyword)  # Keywords
        lexer.setColor(QColor("#569cd6"), QsciLexerPython.ClassName)  # Class names
        lexer.setColor(QColor("#9cdcfe"), QsciLexerPython.FunctionMethodName)  # Function names
        lexer.setColor(QColor("#ce9178"), QsciLexerPython.Comment)  # Comments
        lexer.setColor(QColor("#d69d85"), QsciLexerPython.Number)  # Numbers
        lexer.setColor(QColor("#c586c0"), QsciLexerPython.Decorator)  # Decorators
        lexer.setColor(QColor("#d7ba7d"), QsciLexerPython.SingleQuotedString) # Strings
        lexer.setColor(QColor("#d7ba7d"), QsciLexerPython.DoubleQuotedString)

        # Enable auto-indentation (new lines follow previous indentation)
        self.text_edit.setAutoIndent(True)

        # Configure indentation settings for Python
        self.text_edit.setTabWidth(4)
        self.text_edit.setIndentationWidth(4)
        self.text_edit.setIndentationsUseTabs(False)  # Use spaces instead of tabs
        self.text_edit.setAutoIndent(True)

        # Enable Pythonic behavior: indent after colons (if, for, def, etc.)
        self.text_edit.setIndentationGuides(True)
        self.text_edit.setBraceMatching(QsciScintilla.BraceMatch.SloppyBraceMatch)

        # Set caret (cursor) color
        self.text_edit.setCaretForegroundColor(QColor("#ffffff"))
        
        # Enable word wrap
        self.text_edit.setWrapMode(QsciScintilla.WrapMode.WrapWord)

        # Enable line numbers
        self.text_edit.setMarginType(0, QsciScintilla.MarginType.NumberMargin)
        self.text_edit.setMarginWidth(0, "0000")
        self.text_edit.setMarginsForegroundColor(QColor("#ffffff"))  # Line number color
        self.text_edit.setMarginsBackgroundColor(QColor("#3e3e3e"))  # Line number background

        # Enable brace matching
        self.text_edit.setBraceMatching(QsciScintilla.BraceMatch.SloppyBraceMatch)

        # Enable auto-completion (functions, variables, keywords)
        self.text_edit.setAutoCompletionThreshold(1)  # Show after 1 character
        self.text_edit.setAutoCompletionSource(QsciScintilla.AutoCompletionSource.AcsAPIs)
    
    def load_script(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Script", self.scripts_path, "Python Files (*.py)")
        if file_path:
            try:
                with open(file_path, "r") as file:
                    self.text_edit.setText(file.read())
                self.script_path = file_path
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load script:\n{e}")

    def save_script(self):
        if not self.script_path:
            self.save_script_as()
        else:
            try:
                with open(self.script_path, "w") as file:
                    file.write(self.text_edit.text())
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save script:\n{e}")

    def save_script_as(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Script", self.scripts_path, "Python Files (*.py)")
        if file_path:
            try:
                with open(file_path, "w") as file:
                    file.write(self.text_edit.text())
                self.script_path = file_path
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save script:\n{e}")

    def execute_script(self):
        self.is_executing = True
        self.execution_interrupted = False  # Reset flag before execution

        local_env = self.local_env
        global_env = self.global_env

        def check_interrupt():
            """Checks if execution should be interrupted."""
            QApplication.processEvents()  # Allow for user interaction
            if self.execution_interrupted:
                raise ExecutionInterrupted("Script execution interrupted by user.")

        global_env["check_interrupt"] = check_interrupt # Inject interrupt check into the execution environment
        try:
            # Parse the script and modify its AST
            user_code = self.text_edit.text()
            parsed_ast = ast.parse(user_code)

            # Transform AST to insert check_interrupt()
            transformer = InterruptInjector()
            modified_ast = transformer.visit(parsed_ast)
            ast.fix_missing_locations(modified_ast)
            compiled_code = compile(modified_ast, "<user_script>", "exec")
            exec(compiled_code, global_env, local_env)
        except ExecutionInterrupted:
            QMessageBox.information(self, "Execution Stopped", "Script execution was interrupted.")
        except Exception as e:
            QMessageBox.critical(self, "Execution Error", f"Error:\n{e}")
        finally:
            self.is_executing = False