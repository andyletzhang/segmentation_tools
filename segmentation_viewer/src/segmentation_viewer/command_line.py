from PyQt6.QtCore import pyqtSignal, pyqtSlot, QThread, Qt
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QLineEdit, QMainWindow
from PyQt6.QtGui import QIcon, QTextCursor
import importlib.resources
import io
import sys


class CommandLineWindow(QMainWindow):
    def __init__(self, parent=None, globals_dict={}, locals_dict={}):
        super().__init__()
        icon_path = importlib.resources.files('segmentation_viewer.assets').joinpath('terminal_icon.png')
        self.setWindowIcon(QIcon(str(icon_path)))
        self.setWindowTitle("Command Line")

        # Add the CommandLineWidget to the new window
        self.cli = CommandLineWidget(parent=self, globals_dict=globals_dict, locals_dict=locals_dict)
        self.setCentralWidget(self.cli)  

        # Set the window size and show the window
        self.resize(700, 400)
        self.show()

class CommandLineWidget(QWidget):
    def __init__(self, parent=None, globals_dict={}, locals_dict={}):
        super().__init__(parent)

        # Set up the layout
        layout = QVBoxLayout(self)
        
        # Terminal-style output display area (Read-Only)
        self.terminal_display = QTextEdit(self)
        self.terminal_display.setStyleSheet("""
            background-color: black;
            color: white;
            font-family: "Courier";
            font-size: 10pt;
        """)

        self.terminal_display.setReadOnly(True)
        layout.addWidget(self.terminal_display)
        
        # Command input area
        self.command_input = QLineEdit(self)
        self.command_input.setStyleSheet("""
            background-color: black;
            color: white;
            font-family: "Courier";
            font-size: 12pt;
        """)

        layout.addWidget(self.command_input)

        # Command history
        self.command_history = []
        self.history_index = -1

        # Connect Enter key press to command execution
        self.command_input.returnPressed.connect(self.execute_command)

        self.globals_dict = globals_dict
        self.locals_dict = locals_dict

        # Prompt for commands
        self.prompt = ">>> "

    def showEvent(self, event):
        super().showEvent(event)
        # Set the focus to the command input box when the window is shown
        self.command_input.setFocus()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Up:
            # Navigate through command history (up)
            if self.history_index > 0:
                self.history_index -= 1
                self.command_input.setText(self.command_history[self.history_index])
        elif event.key() == Qt.Key_Down:
            # Navigate through command history (down)
            if self.history_index < len(self.command_history) - 1:
                self.history_index += 1
                self.command_input.setText(self.command_history[self.history_index])
            else:
                self.history_index = len(self.command_history)  # Reset to allow new input
                self.command_input.clear()
        else:
            # Call base class keyPressEvent for default handling
            super().keyPressEvent(event)
    
    def execute_command(self):
        # Get the command from the input box
        command = self.command_input.text()
        self.command_input.clear()

        if command:
            if command in self.command_history: # remove redundant commands further back in the history
                self.command_history.remove(command)
            self.command_history.append(command)
            self.history_index = len(self.command_history)  # Reset index to point to the latest command
            # Display the command in the terminal display
            self.terminal_display.append(self.prompt + command)
            
            # Execute the command and show the result
            self.worker = CodeExecutionWorker(command, self.globals_dict, self.locals_dict)
            self.worker.execution_done.connect(self.on_code_execution_done)
            self.worker.start()
            # block text input until the command is executed
            self.command_input.setDisabled(True)
        self.terminal_display.moveCursor(QTextCursor.End)

    @pyqtSlot(str, str) # Decorator to specify the type of the signal
    def on_code_execution_done(self, output, error):
        if output:
            self.terminal_display.append(output)
            self.terminal_display.moveCursor(QTextCursor.End)
        if error:
            self.terminal_display.append(f"Error: {error}")
            self.terminal_display.moveCursor(QTextCursor.End)
        
        # Re-enable the command input box
        self.command_input.setDisabled(False)
        # Set the focus back to the command input box
        self.command_input.setFocus()

class CodeExecutionWorker(QThread):
    execution_done = pyqtSignal(str, str)  # Signal to emit output and error

    def __init__(self, code, globals_dict, locals_dict):
        super().__init__()
        self.code = code
        self.globals_dict = globals_dict
        self.locals_dict = locals_dict

    def run(self):
        # Create a buffer to capture stdout
        output_buffer = io.StringIO()
        error = ""

        # Redirect stdout to the buffer
        sys_stdout = sys.stdout
        sys.stderr = sys.stderr
        sys.stdout = output_buffer
        sys.stderr = output_buffer

        try:
            # First attempt eval (for expressions)
            output = str(eval(self.code, self.globals_dict, self.locals_dict))
            error = ""

        except SyntaxError:
            # If it’s not an expression, run it as a statement using exec
            try:
                exec(self.code, self.globals_dict, self.locals_dict)
                output = "" # No output for statements
            except Exception as e:
                output = ""
                error = str(e)

        except Exception as e:
            output = ""
            error = str(e)

        finally:
            # Restore stdout
            sys.stdout = sys_stdout
            sys.stderr = sys.stderr

        if not output or output=='None':
            output=output_buffer.getvalue()
        output_buffer.close()

        # Emit the result and any error message back to the main thread
        self.execution_done.emit(output, error)
