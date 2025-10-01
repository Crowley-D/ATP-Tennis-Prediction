"""
UI module for tennis tournament matching system.
Provides GUI dialog interfaces for interactive prompts.
"""

from tennis_matching.ui.gui_prompts import (
    get_dialog_manager,
    cleanup_dialogs,
    TournamentMatchingDialog
)

__all__ = [
    'get_dialog_manager',
    'cleanup_dialogs',
    'TournamentMatchingDialog'
]