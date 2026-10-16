"""Tab registry.

Each feature of the app is one tab: a QWidget subclass with a TAB_TITLE.
To add a feature, create a new module in this folder and add its class to
TABS below - nothing else needs to change.

A tab may implement these optional methods:
    is_busy() -> bool        True while work is running (blocks closing the app)
    shutdown() -> None       Called when the app closes (stop workers, save state)
"""

from app.ui.tabs.converter_tab import ConverterTab
from app.ui.tabs.personalize_tab import PersonalizeTab
from app.ui.tabs.transcribe_tab import TranscribeTab

TABS = [
    ConverterTab,
    TranscribeTab,
    PersonalizeTab,
]
