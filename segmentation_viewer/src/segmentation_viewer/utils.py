from PyQt6.QtGui import QAction


def create_action(name, func, parent=None, shortcut=None):
    action = QAction(name, parent)
    action.triggered.connect(func)
    if shortcut is not None:
        action.setShortcut(shortcut)
    return action


def load_stylesheet(file_path):
    with open(file_path, 'r') as f:
        return f.read()


def create_html_table(labels, values):
    if len(labels) != len(values):
        raise ValueError('Labels and values must be of the same length')

    html = """
    <table style="border-collapse: collapse; width: 100%;">
        <thead>
            <tr>
                <th style="text-align: left; padding: 8px; border-bottom: 2px solid #ddd;">Label</th>
                <th style="text-align: left; padding: 8px; border-bottom: 2px solid #ddd;">Value</th>
            </tr>
        </thead>
        <tbody>
    """
    # Loop to add rows
    for label, value in zip(labels, values):
        value = round(value, 2)  # round to 2 decimal places
        html += f"""
        <tr>
            <td style="padding: 4px;"><b>{label}:</b></td>
            <td style="padding: 4px;">{value}</td>
        </tr>
        """

    html += """
        </tbody>
    </table>
    """
    return html
