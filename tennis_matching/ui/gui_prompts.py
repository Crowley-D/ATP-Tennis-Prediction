"""
GUI prompt wrappers for tournament matching system.
Provides dialog-based user input when running in non-interactive environments.
"""

import tkinter as tk
from tkinter import simpledialog, messagebox
from typing import Optional


class TournamentMatchingDialog:
    """GUI dialog manager for tournament matching prompts."""

    def __init__(self):
        self.root = None
        self._initialize_root()

    def _initialize_root(self):
        """Initialize the Tkinter root window (hidden)."""
        if self.root is None:
            self.root = tk.Tk()
            self.root.withdraw()  # Hide the main window
            # Bring to front on Windows
            self.root.attributes('-topmost', True)
            self.root.update()
            self.root.attributes('-topmost', False)

    def prompt_for_tournament_name(self, suggested_name: str) -> str:
        """
        Prompt user for tournament name via GUI dialog.

        Args:
            suggested_name: Suggested tournament name

        Returns:
            str: User-confirmed tournament name
        """
        self._initialize_root()

        # Force window to front - very aggressive
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()
        self.root.attributes('-topmost', True)
        self.root.update()

        # Schedule to remove topmost after a delay
        self.root.after(100, lambda: self.root.attributes('-topmost', False))

        dialog_msg = f"NEW TOURNAMENT CREATION\n\nSuggested primary name:\n'{suggested_name}'\n\nPress OK to accept, or enter a new name:"

        try:
            result = simpledialog.askstring(
                "New Tournament Name",
                dialog_msg,
                initialvalue=suggested_name,
                parent=self.root
            )
        finally:
            self.root.attributes('-topmost', False)

        # If user cancels or closes dialog, accept suggested name
        if result is None or result.strip() == "":
            return suggested_name

        return result.strip()

    def prompt_for_tournament_id(self, primary_name: str, suggested_id: int,
                                 source_id: str, tourney_level: str) -> int:
        """
        Prompt user for tournament ID via GUI dialog.

        Args:
            primary_name: Primary tournament name
            suggested_id: Suggested tournament ID
            source_id: Original source ID
            tourney_level: Tournament level

        Returns:
            int: User-confirmed tournament ID
        """
        self._initialize_root()

        # Force window to front - very aggressive
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()
        self.root.attributes('-topmost', True)
        self.root.update()

        # Schedule to remove topmost after a delay
        self.root.after(100, lambda: self.root.attributes('-topmost', False))

        dialog_msg = (
            f"TOURNAMENT ID SELECTION\n\n"
            f"Tournament: {primary_name}\n"
            f"Level: {tourney_level}\n"
            f"Source ID: {source_id}\n\n"
            f"Suggested ID: {suggested_id}\n\n"
            f"Press OK to accept, or enter a different ID:"
        )

        try:
            result = simpledialog.askinteger(
                "Tournament ID Selection",
                dialog_msg,
                initialvalue=suggested_id,
                minvalue=1,
                maxvalue=9999,
                parent=self.root
            )
        finally:
            self.root.attributes('-topmost', False)

        # If user cancels, accept suggested ID
        if result is None:
            return suggested_id

        return result

    def confirm_high_similarity_match(self, target_name: str, matched_name: str,
                                     similarity: float, source_name: str) -> bool:
        """
        Confirm a high-similarity match via GUI dialog.

        Args:
            target_name: Target tournament name
            matched_name: Matched tournament name from database
            similarity: Similarity score (0-1)
            source_name: Source name for context

        Returns:
            bool: True if user confirms match, False otherwise
        """
        print(f"[GUI DEBUG] confirm_high_similarity_match called")
        self._initialize_root()

        # Force window to front - very aggressive
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()
        self.root.attributes('-topmost', True)
        self.root.update()

        dialog_msg = (
            f"HIGH SIMILARITY MATCH\n\n"
            f"Your tournament: '{target_name}'\n"
            f"Matched with: '{matched_name}'\n"
            f"Similarity: {similarity:.1%}\n"
            f"Source: {source_name}\n\n"
            f"Accept this match?"
        )

        try:
            result = messagebox.askyesno(
                "Confirm Tournament Match",
                dialog_msg,
                parent=self.root
            )
            print(f"[GUI DEBUG] User response: {result}")
        finally:
            self.root.attributes('-topmost', False)

        return result

    def prompt_for_match_confirmation(self, target_name: str, matches: list,
                                     source_name: str, source_id: str) -> tuple:
        """
        Prompt user to select from multiple tournament matches via GUI dialog.

        Args:
            target_name: Target tournament name
            matches: List of potential matches with similarity scores
            source_name: Source name for context
            source_id: Source ID for context

        Returns:
            tuple: (selected_tournament_id or None, action_taken)
        """
        self._initialize_root()

        # Force root window to front first
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()
        self.root.attributes('-topmost', True)
        self.root.update()

        # Create a custom dialog window
        dialog = tk.Toplevel(self.root)
        dialog.title("Tournament Match Review")
        dialog.geometry("600x500")

        # Force dialog to front
        dialog.lift()
        dialog.focus_force()
        dialog.attributes('-topmost', True)
        dialog.update()

        # Make it modal
        dialog.transient(self.root)
        dialog.grab_set()

        # Header
        header_frame = tk.Frame(dialog, bg="#2c3e50", pady=10)
        header_frame.pack(fill=tk.X)

        tk.Label(
            header_frame,
            text="TOURNAMENT MATCH REVIEW",
            font=("Arial", 14, "bold"),
            bg="#2c3e50",
            fg="white"
        ).pack()

        # Info section
        info_frame = tk.Frame(dialog, pady=10)
        info_frame.pack(fill=tk.X, padx=20)

        tk.Label(info_frame, text=f"Source: {source_name}", anchor="w").pack(fill=tk.X)
        tk.Label(info_frame, text=f"Source ID: {source_id}", anchor="w").pack(fill=tk.X)
        tk.Label(info_frame, text=f"Tournament: '{target_name}'",
                font=("Arial", 10, "bold"), anchor="w").pack(fill=tk.X, pady=5)

        # Separator
        tk.Frame(dialog, height=2, bg="#bdc3c7").pack(fill=tk.X, padx=20)

        # Matches list
        matches_frame = tk.Frame(dialog, pady=10)
        matches_frame.pack(fill=tk.BOTH, expand=True, padx=20)

        tk.Label(matches_frame, text="Potential matches:",
                font=("Arial", 10, "bold")).pack(anchor="w", pady=5)

        # Scrollable listbox
        listbox_frame = tk.Frame(matches_frame)
        listbox_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(listbox_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        listbox = tk.Listbox(listbox_frame, yscrollcommand=scrollbar.set,
                            font=("Courier", 9), height=10)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)

        # Populate matches
        for i, match in enumerate(matches, 1):
            similarity = match['similarity_score']
            existing_name = match['source_name_variant']
            existing_source = match['source_name']
            tournament_id = match['tournament_id']

            listbox.insert(tk.END, f"{i}. '{existing_name}' (ID: {tournament_id}, {existing_source}) - {similarity:.1%}")

        # Buttons
        button_frame = tk.Frame(dialog, pady=10)
        button_frame.pack(fill=tk.X, padx=20)

        result = {'tournament_id': None, 'action': 'user_cancelled'}

        def on_select_match():
            selection = listbox.curselection()
            if selection:
                idx = selection[0]
                selected_match = matches[idx]
                result['tournament_id'] = selected_match['tournament_id']
                result['action'] = 'manual_confirmation'
                dialog.destroy()
            else:
                messagebox.showwarning("No Selection", "Please select a match first")

        def on_create_new():
            result['tournament_id'] = None
            result['action'] = 'create_new_tournament'
            dialog.destroy()

        def on_skip():
            result['tournament_id'] = None
            result['action'] = 'manual_skip'
            dialog.destroy()

        tk.Button(button_frame, text="Select Match", command=on_select_match,
                 bg="#27ae60", fg="white", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Create New Tournament", command=on_create_new,
                 bg="#3498db", fg="white", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Skip", command=on_skip,
                 bg="#e74c3c", fg="white", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)

        # Wait for dialog to close
        dialog.wait_window()

        return result['tournament_id'], result['action']

    def prompt_for_edited_name_matches(self, edited_name: str, matches: list) -> Optional[str]:
        """
        Prompt user when edited tournament name matches existing tournaments.

        Args:
            edited_name: The name user entered
            matches: List of similar tournament matches (>= 60%)

        Returns:
            str: Special marker string with tournament ID if user selects existing tournament
            None: If user wants to proceed with new tournament
        """
        print(f"[GUI DEBUG] prompt_for_edited_name_matches called with {len(matches)} matches")
        self._initialize_root()
        print(f"[GUI DEBUG] Root initialized")

        # Force root window to front first
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()
        self.root.attributes('-topmost', True)
        self.root.update()

        # Create a custom dialog window
        dialog = tk.Toplevel(self.root)
        print(f"[GUI DEBUG] Toplevel created")
        dialog.title("Similar Tournaments Found")
        dialog.geometry("650x550")

        # Force dialog to front
        dialog.lift()
        dialog.focus_force()
        dialog.attributes('-topmost', True)
        dialog.update()

        # Make it modal
        dialog.transient(self.root)
        dialog.grab_set()

        # Header
        header_frame = tk.Frame(dialog, bg="#e67e22", pady=10)
        header_frame.pack(fill=tk.X)

        tk.Label(
            header_frame,
            text="SIMILAR TOURNAMENTS FOUND",
            font=("Arial", 14, "bold"),
            bg="#e67e22",
            fg="white"
        ).pack()

        # Info section
        info_frame = tk.Frame(dialog, pady=10)
        info_frame.pack(fill=tk.X, padx=20)

        tk.Label(info_frame, text=f"Your edited name: '{edited_name}'",
                font=("Arial", 10, "bold"), anchor="w", fg="#e67e22").pack(fill=tk.X, pady=5)
        tk.Label(info_frame, text=f"Found {len(matches)} similar tournament(s) in the database:",
                anchor="w").pack(fill=tk.X)

        # Separator
        tk.Frame(dialog, height=2, bg="#bdc3c7").pack(fill=tk.X, padx=20, pady=5)

        # Matches list
        matches_frame = tk.Frame(dialog, pady=10)
        matches_frame.pack(fill=tk.BOTH, expand=True, padx=20)

        tk.Label(matches_frame, text="Similar tournaments:",
                font=("Arial", 10, "bold")).pack(anchor="w", pady=5)

        # Scrollable listbox
        listbox_frame = tk.Frame(matches_frame)
        listbox_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(listbox_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        listbox = tk.Listbox(listbox_frame, yscrollcommand=scrollbar.set,
                            font=("Courier", 9), height=12)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)

        # Populate matches
        for i, match in enumerate(matches[:10], 1):  # Show up to 10
            similarity = match['similarity_score']
            primary_name = match['primary_name']
            tournament_id = match['tournament_id']
            source_variant = match['source_name_variant']
            level = match['tourney_level']

            listbox.insert(tk.END, f"{i}. '{primary_name}' (ID: {tournament_id}, Level: {level})")
            listbox.insert(tk.END, f"   Matched: '{source_variant}' - {similarity:.1%}")
            listbox.insert(tk.END, "")  # Empty line for spacing

        # Buttons
        button_frame = tk.Frame(dialog, pady=10)
        button_frame.pack(fill=tk.X, padx=20)

        result = {'value': None}

        def on_select_match():
            selection = listbox.curselection()
            if selection:
                # Each match takes 3 lines (name, matched, empty), so divide by 3
                idx = selection[0] // 3
                if idx < len(matches):
                    selected_match = matches[idx]
                    result['value'] = f"__USE_EXISTING_TOURNAMENT__:{selected_match['tournament_id']}"
                    dialog.destroy()
            else:
                messagebox.showwarning("No Selection", "Please select a tournament first")

        def on_create_new():
            result['value'] = None  # Signal to create new tournament
            dialog.destroy()

        tk.Button(button_frame, text="Use Selected Tournament", command=on_select_match,
                 bg="#27ae60", fg="white", font=("Arial", 10, "bold"), width=20).pack(pady=5)
        tk.Button(button_frame, text="Proceed with New Tournament", command=on_create_new,
                 bg="#3498db", fg="white", font=("Arial", 10), width=20).pack(pady=5)

        # Wait for dialog to close
        dialog.wait_window()

        return result['value']

    def cleanup(self):
        """Clean up the Tkinter root window."""
        if self.root:
            try:
                self.root.destroy()
            except:
                pass
            self.root = None


# Global singleton instance
_dialog_manager = None


def get_dialog_manager() -> TournamentMatchingDialog:
    """Get or create the global dialog manager instance."""
    global _dialog_manager
    if _dialog_manager is None:
        _dialog_manager = TournamentMatchingDialog()
    return _dialog_manager


def cleanup_dialogs():
    """Clean up the dialog manager."""
    global _dialog_manager
    if _dialog_manager:
        _dialog_manager.cleanup()
        _dialog_manager = None