# GUI Dialogs for Tournament Matching

## Overview

The tournament matching system now supports **GUI dialog windows** for interactive prompts instead of terminal input. This is especially useful when running the pipeline from scripts or IDEs where terminal `input()` calls fail with EOFError.

## Features

✅ **Separate dialog windows** for all user prompts
✅ **Automatic fallback** to terminal if GUI unavailable
✅ **Clean, modern interface** with yes/no buttons and selection lists
✅ **No code changes required** - works automatically

## How It Works

### Automatic Mode (Default)

GUI dialogs are **enabled by default**. When you run the pipeline:

```python
from Main_Matches_Dataset_Pipeline import process_matches_pipeline

input_df = pd.read_csv("./data/PreCleanedMatches/JeffSackmann.csv")
process_matches_pipeline(input_df, replace=False, backup_existing=False)
```

**GUI dialog windows will automatically appear** for:
- ✅ Tournament name confirmation
- ✅ Tournament ID selection
- ✅ High similarity match confirmation
- ✅ Multi-match selection dialogs

### Disable GUI (Use Terminal)

To disable GUI and use terminal input:

```python
import os
os.environ['TOURNAMENT_MATCHING_GUI'] = '0'

# Now run your pipeline - will use terminal input
```

Or set the environment variable before running Python:

**Windows (PowerShell)**:
```powershell
$env:TOURNAMENT_MATCHING_GUI="0"
python Main_Matches_Dataset_Pipeline.py
```

**Windows (CMD)**:
```cmd
set TOURNAMENT_MATCHING_GUI=0
python Main_Matches_Dataset_Pipeline.py
```

**Linux/Mac**:
```bash
export TOURNAMENT_MATCHING_GUI=0
python Main_Matches_Dataset_Pipeline.py
```

## Dialog Types

### 1. Tournament Name Dialog
<img src="..." alt="Name dialog" width="500"/>

**When shown**: Creating a new tournament
**Purpose**: Confirm or edit the tournament name
**Actions**:
- Click **OK** or press **Enter** to accept suggested name
- **Type a new name** and click OK to use custom name
- Click **Cancel** to use suggested name (fallback)

### 2. Tournament ID Dialog
<img src="..." alt="ID dialog" width="500"/>

**When shown**: Creating a new tournament (after name confirmation)
**Purpose**: Confirm or edit the tournament ID
**Actions**:
- Click **OK** to accept suggested ID
- **Enter a custom 4-digit ID** and click OK
- Click **Cancel** to use suggested ID (fallback)

### 3. High Similarity Match Dialog
<img src="..." alt="Similarity dialog" width="500"/>

**When shown**: Found a close match (80-99.4% similarity)
**Purpose**: Confirm if two tournament names refer to the same tournament
**Actions**:
- Click **Yes** to accept the match
- Click **No** to reject and create new tournament

### 4. Multi-Match Selection Dialog
<img src="..." alt="Multi-match dialog" width="600"/>

**When shown**: Multiple potential matches found (60-79% similarity)
**Purpose**: Select the correct match or create new tournament
**Actions**:
- **Select a match** from the list and click "Select Match"
- Click **"Create New Tournament"** if none match
- Click **"Skip"** to postpone decision (tournament gets null ID)

## Troubleshooting

### "GUI dialogs not available, using terminal input"

**Cause**: Tkinter is not installed or not accessible

**Solution**:
- Tkinter usually comes with Python
- Try: `python -m tkinter` to test if it's available
- Reinstall Python with "tcl/tk" option enabled

### Dialogs appear but then freeze

**Cause**: Main thread blocking or Tkinter event loop conflict

**Solution**:
- Make sure you're not running inside another GUI application
- Try running from command line instead of IDE
- Set `TOURNAMENT_MATCHING_GUI=0` to disable GUI

### No dialogs appear at all

**Cause**: Running in a truly headless environment (no display)

**Solution**:
- The system will automatically fall back to terminal input
- Or set `TOURNAMENT_MATCHING_GUI=0` explicitly

## Technical Details

### Implementation

- GUI uses Python's built-in `tkinter` library
- Dialogs are modal (block until user responds)
- Singleton pattern ensures one dialog manager per session
- Automatic cleanup on exit

### Fallback Behavior

If GUI fails for any reason:
1. Catches exception
2. Prints warning message
3. Falls back to terminal `input()`
4. If terminal also fails (EOFError), auto-accepts defaults

This ensures the pipeline **never crashes** due to input issues.

### EOFError Handling

Even with GUI disabled, all `input()` calls now have EOFError handling:

```python
try:
    choice = input("Enter choice: ")
except EOFError:
    # Auto-accept default value
    choice = default_value
```

This means the pipeline can run **completely non-interactively** if needed.

## Examples

### Run with GUI (default)
```python
python Main_Matches_Dataset_Pipeline.py
```
→ GUI dialogs appear in separate windows

### Run in terminal mode
```python
os.environ['TOURNAMENT_MATCHING_GUI'] = '0'
python Main_Matches_Dataset_Pipeline.py
```
→ Prompts appear in terminal/console

### Fully automated (no prompts)
Coming soon: `auto_resolve=True` parameter

## Benefits

✅ **Better UX**: Cleaner interface than terminal prompts
✅ **Works in IDEs**: No more EOFError when running from PyCharm/VS Code
✅ **Non-blocking**: Continue working while dialogs are open
✅ **Robust**: Multiple fallback layers ensure reliability
✅ **Zero config**: Works out of the box

## Support

If you encounter issues with GUI dialogs:
1. Check if Tkinter is installed: `python -m tkinter`
2. Try terminal mode: `TOURNAMENT_MATCHING_GUI=0`
3. Check console output for warning messages
4. Report issues with full error traceback