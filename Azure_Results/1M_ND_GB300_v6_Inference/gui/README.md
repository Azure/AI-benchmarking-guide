# Modular Curses Form System

A clean, modular curses-based form system for Python applications. This system provides reusable components for building interactive text-based user interfaces.

## Features

- **Modular Design**: Each form section is a separate class with well-defined interfaces
- **Multiple Section Types**: Text display, text input, single selection, multi-selection, and sub-forms
- **Keyboard Navigation**: Tab/Shift+Tab navigation, arrow key support
- **Flexible Layout**: Automatic scrolling, alignment options
- **Easy Integration**: Simple API for creating and running forms

## Section Types

### 1. TextDisplaySection
Non-interactive text display with alignment options.

```python
from window import TextDisplaySection, Alignment

section = TextDisplaySection(
    text="Welcome to the application\nThis is a multi-line display",
    alignment=Alignment.CENTER  # LEFT, CENTER, or RIGHT
)
```

### 2. TextInputSection
Text input field with cursor support and editing capabilities.

```python
from window import TextInputSection

# Single-line input (newlines not allowed)
section = TextInputSection(
    prompt="Enter your name:",
    default_value="Anonymous",
    allow_newlines=False
)

# Multi-line input (newlines allowed)
section = TextInputSection(
    prompt="Enter description:",
    default_value="Line 1\nLine 2",
    allow_newlines=True  # This is the default
)
```

**Features:**
- Full cursor navigation (arrow keys, home, end)
- Backspace and delete support
- Real-time text highlighting when active
- **Configurable newline handling**:
  - `allow_newlines=True` (default): Enter key adds newlines for multi-line text
  - `allow_newlines=False`: Enter key submits form, creating single-line inputs

### 3. SingleSelectSection
Radio button-style single selection from a list.

```python
from window import SingleSelectSection

section = SingleSelectSection(
    prompt="Choose your favorite color:",
    options=["Red", "Green", "Blue", "Yellow"],
    value_formatter=lambda x: f"🎨 {x}"  # Optional custom formatting
)
```

**Controls:**
- Up/Down arrows to navigate
- Space to select (for consistency, though selection follows cursor)

### 4. MultiSelectSection
Checkbox-style multiple selection from a list.

```python
from window import MultiSelectSection

section = MultiSelectSection(
    prompt="Select your hobbies:",
    options=["Reading", "Gaming", "Sports", "Music"],
    value_formatter=lambda x: f"⭐ {x}"
)
```

**Controls:**
- Up/Down arrows to navigate
- Space to toggle selection of current item

### 5. SubFormSection
Button that opens another form as a modal dialog.

```python
from window import SubFormSection

# Define sub-form sections
sub_sections = {
    'name': TextInputSection("Name:", ""),
    'priority': SingleSelectSection("Priority:", ["Low", "High"])
}

section = SubFormSection(
    button_text="Advanced Settings",
    form_title="Configuration",
    form_sections=sub_sections
)
```

## Creating and Running Forms

### Basic Usage

```python
from window import run_form, TextDisplaySection, TextInputSection

# Define form sections
sections = {
    'welcome': TextDisplaySection("Welcome to the form!"),
    'name': TextInputSection("Your name:", ""),
    'email': TextInputSection("Your email:", "")
}

# Run the form
result = run_form("User Information", sections)

if result:
    print(f"Name: {result['name']}")
    print(f"Email: {result['email']}")
else:
    print("Form cancelled")
```

### Advanced Usage

```python
from window import *

def create_main_form():
    # Sub-form for advanced settings
    advanced_sections = {
        'debug': SingleSelectSection("Debug level:", ["None", "Info", "Debug"]),
        'timeout': TextInputSection("Timeout (seconds):", "30")
    }

    # Main form sections
    sections = {
        'header': TextDisplaySection(
            "Application Configuration\nFill out the form below",
            alignment=Alignment.CENTER
        ),
        'app_name': TextInputSection("Application name:", "MyApp"),
        'features': MultiSelectSection(
            "Enable features:",
            ["Logging", "Authentication", "Caching", "Monitoring"]
        ),
        'environment': SingleSelectSection(
            "Target environment:",
            ["Development", "Staging", "Production"]
        ),
        'advanced': SubFormSection(
            "Advanced Settings",
            "Advanced Configuration",
            advanced_sections
        )
    }

    return run_form("App Configuration", sections)

# Run the form
config = create_main_form()
if config:
    print("Configuration saved:", config)
```

## Keyboard Controls

### Global Controls
- **Tab**: Move to next section
- **Shift+Tab**: Move to previous section
- **Enter**: Submit form (when not in text input)
- **Escape**: Cancel form

### Section-Specific Controls

#### TextInputSection
- **Arrow Keys**: Move cursor left/right
- **Home/End**: Jump to beginning/end of text
- **Backspace/Delete**: Remove characters
- **Enter**: Add newline (if `allow_newlines=True`) or submit form (if `allow_newlines=False`)
- **Printable characters**: Insert text at cursor

#### SingleSelectSection & MultiSelectSection
- **Up/Down Arrows**: Navigate options
- **Space**: Select option (SingleSelect) or toggle selection (MultiSelect)

#### SubFormSection
- **Space/Enter**: Open sub-form

## API Reference

### Section Base Class

All sections inherit from the `Section` abstract base class:

```python
class Section(ABC):
    @abstractmethod
    def draw(self, stdscr, y: int, width: int) -> int:
        """Draw the section and return height used"""

    @abstractmethod
    def handle_key(self, key: int) -> bool:
        """Handle key input, return True if handled"""

    @abstractmethod
    def get_value(self) -> Any:
        """Get current value of the section"""

    def get_height(self) -> int:
        """Get height this section occupies"""
```

### Form Class

```python
class Form:
    def __init__(self, title: str, sections: Dict[str, Section]):
        """Create a form with given title and sections"""

    def run(self, stdscr) -> Optional[Dict[str, Any]]:
        """Run the form interactively, return values or None if cancelled"""

    def get_value(self) -> Dict[str, Any]:
        """Get current values from all sections"""
```

### Convenience Function

```python
def run_form(title: str, sections: Dict[str, Section]) -> Optional[Dict[str, Any]]:
    """Run a form in a curses environment (convenience wrapper)"""
```

## Examples

See `example.py` for a comprehensive demonstration of all section types and features.

Run the example:
```bash
python3 example.py
```

Run basic tests:
```bash
python3 test_basic.py
```

## Design Principles

1. **Modularity**: Each section type is independent and reusable
2. **Separation of Concerns**: Drawing, input handling, and value management are clearly separated
3. **Consistent Interface**: All sections follow the same API pattern
4. **Extensibility**: Easy to add new section types by inheriting from `Section`
5. **Robustness**: Handles terminal resize, drawing errors, and edge cases gracefully

## Extending the System

To create a custom section type:

```python
class CustomSection(Section):
    def __init__(self, custom_param):
        super().__init__()
        self.custom_param = custom_param

    def draw(self, stdscr, y: int, width: int) -> int:
        # Draw your custom section
        # Return height used
        pass

    def handle_key(self, key: int) -> bool:
        # Handle key input for your section
        # Return True if key was handled
        pass

    def get_value(self) -> Any:
        # Return the current value
        pass
```

## Requirements

- Python 3.6+
- Standard library `curses` module (included on Unix-like systems)
- Terminal that supports curses (most terminals do)

## License

Part of the MLPerf inference benchmark suite.