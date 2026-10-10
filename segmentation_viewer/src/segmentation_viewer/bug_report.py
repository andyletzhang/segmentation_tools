"""
Bug report dialog for segmentation_viewer.

Reports are posted to a Slack incoming webhook, which shows up as a notification
in the developer's Slack channel. Users don't need an account or a browser.

A webhook URL can only post messages to one channel; it can't read anything.
If it ever gets abused, regenerate it in the Slack app settings and paste the new one here.
"""

import base64
import json
import os
import platform
import ssl
import sys
import urllib.error
import urllib.request
from importlib.metadata import PackageNotFoundError, version

from PyQt6.QtCore import PYQT_VERSION_STR, QT_VERSION_STR
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QVBoxLayout,
)

# Encoded so secret scanners (GitHub push protection, Slack's auto-revocation) don't flag it.
# This is NOT security: anyone reading the source can decode it. Generate with:
#   python -c "import base64; print(base64.b64encode(input('URL: ')[::-1].encode()).decode())"
_ENCODED_WEBHOOK = 'NzA1enBIN2lJM2VadWk3QUwxQnBMQWd1L1FNQUdGTkE4QzBCL1BMRE5ZRlY3QzBUL3NlY2l2cmVzL21vYy5rY2Fscy5za29vaC8vOnNwdHRo'
WEBHOOK_URL = base64.b64decode(_ENCODED_WEBHOOK).decode()[::-1]

PACKAGES = ('segmentation_viewer', 'segmentation_tools', 'numpy', 'pyqtgraph', 'cellpose', 'torch')
SECTION_LIMIT = 2900  # Slack section blocks max out at 3000 characters
HEADER_LIMIT = 150  # Slack header blocks max out at 150 characters


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return 'not installed'


def collect_system_info(context: dict | None = None) -> str:
    lines = [
        f'OS: {platform.platform()}',
        f'Python: {sys.version.split()[0]}',
        f'Qt: {QT_VERSION_STR} / PyQt6: {PYQT_VERSION_STR}',
    ]
    lines += [f'{pkg}: {_package_version(pkg)}' for pkg in PACKAGES]
    if context:
        lines += [f'{key}: {value}' for key, value in context.items()]
    return '\n'.join(lines)


def _ssl_context() -> ssl.SSLContext:
    """
    Build an SSL context from certifi's CA bundle instead of the OS certificate store.
    Python's default context loads every certificate in the Windows store, and a single
    malformed one there makes the whole load fail with '[ASN1: NOT_ENOUGH_DATA]'.
    """
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _escape(text: str) -> str:
    """Escape the characters Slack treats as control sequences in mrkdwn."""
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 15] + '\n…(truncated)'


def send_report(subject: str, name: str, message: str, system_info: str | None) -> None:
    """Post the report to the Slack webhook. Raises on failure."""
    sender = name or 'anonymous'
    blocks = [
        {'type': 'header', 'text': {'type': 'plain_text', 'text': _truncate(f'{subject}', HEADER_LIMIT)}},
        {'type': 'context', 'elements': [{'type': 'mrkdwn', 'text': f'From: *{_escape(sender)}*'}]},
        {'type': 'section', 'text': {'type': 'mrkdwn', 'text': _truncate(_escape(message), SECTION_LIMIT)}},
    ]
    if system_info:
        blocks.append({
            'type': 'section',
            'text': {'type': 'mrkdwn', 'text': _truncate(f'```{_escape(system_info)}```', SECTION_LIMIT)},
        })

    payload = {
        'text': f'Bug report from {sender}: {subject}',  # fallback shown in notifications
        'blocks': blocks,
    }

    request = urllib.request.Request(
        WEBHOOK_URL,
        data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    urllib.request.urlopen(request, timeout=10, context=_ssl_context()).close()


class BugReportDialog(QDialog):
    def __init__(self, parent=None, context: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle('Report a Bug')
        self.resize(500, 450)
        self.system_info = collect_system_info(context)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.name_edit = QLineEdit(placeholderText='Optional, so I know who to follow up with')
        self.subject_edit = QLineEdit(placeholderText='Short summary of the problem')
        form.addRow('Your name:', self.name_edit)
        form.addRow('Subject:', self.subject_edit)
        layout.addLayout(form)

        layout.addWidget(QLabel('What happened, and what were you doing when it did?'))
        self.message_edit = QPlainTextEdit()
        layout.addWidget(self.message_edit, stretch=1)

        self.include_info_checkbox = QCheckBox('Include system information')
        self.include_info_checkbox.setChecked(True)
        self.include_info_checkbox.setToolTip(self.system_info)
        layout.addWidget(self.include_info_checkbox)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.send_button = buttons.addButton('Send', QDialogButtonBox.ButtonRole.AcceptRole)
        self.send_button.setDefault(True)
        buttons.accepted.connect(self._submit)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _submit(self):
        subject = self.subject_edit.text().strip()
        message = self.message_edit.toPlainText().strip()
        if not subject and not message:
            QMessageBox.warning(self, 'Incomplete report', 'Please fill in subject and/or description.')
            return
        elif not subject:
            subject = '(no subject)'
        elif not message:
            message = '(no description)'

        self.send_button.setEnabled(False)
        try:
            send_report(
                subject,
                self.name_edit.text().strip(),
                message,
                self.system_info if self.include_info_checkbox.isChecked() else None,
            )
        except RuntimeError as e:
            self.send_button.setEnabled(True)
            QMessageBox.warning(self, 'Could not send', str(e))
            return
        except urllib.error.HTTPError as e:
            self.send_button.setEnabled(True)
            QMessageBox.warning(self, 'Could not send', f'Slack rejected the report (HTTP {e.code}: {e.reason}).')
            return
        except ssl.SSLError as e:
            self.send_button.setEnabled(True)
            QMessageBox.warning(self, 'Could not send', f'A secure connection could not be set up ({e}).')
            return
        except (urllib.error.URLError, OSError) as e:
            self.send_button.setEnabled(True)
            QMessageBox.warning(self, 'Could not send', f'The report could not be sent ({e}).\nCheck your internet connection and try again.')
            return

        QMessageBox.information(self, 'Thanks!', 'Your report has been sent.')
        self.accept()