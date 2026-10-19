import curses
import os
from typing import List, Optional, Set, Dict, Any, Callable
from abc import ABC, abstractmethod
from enum import Enum


class Alignment(Enum):
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


class Section(ABC):
    """Base class for all form sections"""

    def __init__(self):
        self.is_active = False

    @abstractmethod
    def draw(self, stdscr, y: int, width: int) -> int:
        """Draw the section at position y and return the height used"""
        pass

    @abstractmethod
    def handle_key(self, key: int) -> bool:
        """Handle key input. Return True if key was handled."""
        pass

    @abstractmethod
    def get_value(self) -> Any:
        """Get the current value of this section"""
        pass

    def get_height(self) -> int:
        """Get the height this section will occupy"""
        return 1


class TextDisplaySection(Section):
    """Non-interactive text display with alignment options"""

    def __init__(self, text: str, alignment: Alignment = Alignment.LEFT):
        super().__init__()
        self.text = text
        self.alignment = alignment

    def draw(self, stdscr, y: int, width: int) -> int:
        lines = self.text.split('\n')
        for i, line in enumerate(lines):
            if self.alignment == Alignment.LEFT:
                x = 0
            elif self.alignment == Alignment.CENTER:
                x = max(0, (width - len(line)) // 2)
            else:  # RIGHT
                x = max(0, width - len(line))

            try:
                stdscr.addstr(y + i, x, line)
            except curses.error:
                pass  # Ignore if we can't draw at this position

        return len(lines)

    def handle_key(self, key: int) -> bool:
        return False  # Text displays don't handle keys

    def get_value(self) -> str:
        return self.text

    def get_height(self) -> int:
        return len(self.text.split('\n'))


class TextInputSection(Section):
    """Text input with highlighting"""

    def __init__(self, prompt: str, default_value: str = "", allow_newlines: bool = True):
        super().__init__()
        self.prompt = prompt
        self.value = default_value
        self.cursor_pos = len(default_value)
        self.allow_newlines = allow_newlines

    def draw(self, stdscr, y: int, width: int) -> int:
        # Draw prompt
        try:
            stdscr.addstr(y, 0, self.prompt, curses.A_BOLD)
        except curses.error:
            pass

        # Draw input field
        input_y = y + 1
        input_text = self.value

        # Show cursor if active
        if self.is_active:
            display_text = input_text[:self.cursor_pos] + "█" + input_text[self.cursor_pos:]
            attr = curses.color_pair(1) | curses.A_BOLD
        else:
            display_text = input_text
            attr = curses.color_pair(2)

        # Handle multi-line text if newlines are allowed
        if self.allow_newlines and '\n' in display_text:
            lines = display_text.split('\n')
            for i, line in enumerate(lines):
                try:
                    stdscr.addstr(input_y + i, 2, line[:width-3], attr)
                except curses.error:
                    pass
            return 1 + len(lines)  # prompt + number of text lines
        else:
            # Single line display
            try:
                stdscr.addstr(input_y, 2, display_text[:width-3], attr)
            except curses.error:
                pass
            return 2  # prompt + single input line

    def handle_key(self, key: int) -> bool:
        if not self.is_active:
            return False

        if key == curses.KEY_BACKSPACE or key == 127:  # Backspace
            if self.cursor_pos > 0:
                self.value = self.value[:self.cursor_pos-1] + self.value[self.cursor_pos:]
                self.cursor_pos -= 1
            return True
        elif key == 330:  # Delete key (may not be available as curses.KEY_DELETE on all systems)
            if self.cursor_pos < len(self.value):
                self.value = self.value[:self.cursor_pos] + self.value[self.cursor_pos+1:]
            return True
        elif key == curses.KEY_LEFT:
            self.cursor_pos = max(0, self.cursor_pos - 1)
            return True
        elif key == curses.KEY_RIGHT:
            self.cursor_pos = min(len(self.value), self.cursor_pos + 1)
            return True
        elif key == curses.KEY_HOME:
            self.cursor_pos = 0
            return True
        elif key == curses.KEY_END:
            self.cursor_pos = len(self.value)
            return True
        elif key == 10 or key == 13:  # Enter/Return
            if self.allow_newlines:
                self.value = self.value[:self.cursor_pos] + '\n' + self.value[self.cursor_pos:]
                self.cursor_pos += 1
                return True
            else:
                # If newlines not allowed, ignore the key (don't handle it)
                # This lets the form handle it as a submit action
                return False
        elif 32 <= key <= 126:  # Printable characters
            self.value = self.value[:self.cursor_pos] + chr(key) + self.value[self.cursor_pos:]
            self.cursor_pos += 1
            return True

        return False

    def get_value(self) -> str:
        return self.value

    def get_height(self) -> int:
        if self.allow_newlines and '\n' in self.value:
            return 1 + len(self.value.split('\n'))  # prompt + number of text lines
        else:
            return 2  # prompt + single input line


class SingleSelectSection(Section):
    """Radio button list for single selection"""

    def __init__(self, prompt: str, options: List[Any], value_formatter: Callable[[Any], str] = str):
        super().__init__()
        self.prompt = prompt
        self.options = options
        self.value_formatter = value_formatter
        self.selected_index = 0

    def draw(self, stdscr, y: int, width: int) -> int:
        # Draw prompt
        try:
            stdscr.addstr(y, 0, self.prompt, curses.A_BOLD)
        except curses.error:
            pass

        # Draw options
        for i, option in enumerate(self.options):
            option_y = y + 1 + i
            marker = "●" if i == self.selected_index else "○"
            text = f"{marker} {self.value_formatter(option)}"

            if self.is_active and i == self.selected_index:
                attr = curses.color_pair(1) | curses.A_BOLD
            else:
                attr = curses.color_pair(2)

            try:
                stdscr.addstr(option_y, 2, text[:width-3], attr)
            except curses.error:
                pass

        return len(self.options) + 1

    def handle_key(self, key: int) -> bool:
        if not self.is_active:
            return False

        if key == curses.KEY_UP:
            self.selected_index = (self.selected_index - 1) % len(self.options)
            return True
        elif key == curses.KEY_DOWN:
            self.selected_index = (self.selected_index + 1) % len(self.options)
            return True
        elif key == ord(' '):
            # Space doesn't change selection in radio buttons, just for consistency
            return True

        return False

    def get_value(self) -> Any:
        return self.options[self.selected_index] if self.options else None

    def get_height(self) -> int:
        return len(self.options) + 1


class MultiSelectSection(Section):
    """Checkbox list for multiple selection"""

    def __init__(self, prompt: str, options: List[Any], value_formatter: Callable[[Any], str] = str):
        super().__init__()
        self.prompt = prompt
        self.options = options
        self.value_formatter = value_formatter
        self.selected_indices: Set[int] = set()
        self.current_index = 0

    def draw(self, stdscr, y: int, width: int) -> int:
        # Draw prompt
        try:
            stdscr.addstr(y, 0, self.prompt, curses.A_BOLD)
        except curses.error:
            pass

        # Draw options
        for i, option in enumerate(self.options):
            option_y = y + 1 + i
            marker = "●" if i in self.selected_indices else "○"
            text = f"{marker} {self.value_formatter(option)}"

            if self.is_active and i == self.current_index:
                attr = curses.color_pair(1) | curses.A_BOLD
            else:
                attr = curses.color_pair(2)

            try:
                stdscr.addstr(option_y, 2, text[:width-3], attr)
            except curses.error:
                pass

        return len(self.options) + 1

    def handle_key(self, key: int) -> bool:
        if not self.is_active:
            return False

        if key == curses.KEY_UP:
            self.current_index = (self.current_index - 1) % len(self.options)
            return True
        elif key == curses.KEY_DOWN:
            self.current_index = (self.current_index + 1) % len(self.options)
            return True
        elif key == ord(' '):
            if self.current_index in self.selected_indices:
                self.selected_indices.remove(self.current_index)
            else:
                self.selected_indices.add(self.current_index)
            return True

        return False

    def get_value(self) -> List[Any]:
        return [self.options[i] for i in sorted(self.selected_indices)]

    def get_height(self) -> int:
        return len(self.options) + 1


class SubFormSection(Section):
    """Button that opens a sub-form"""

    def __init__(self, button_text: str, form_title: str, form_sections: Dict[str, Section]):
        super().__init__()
        self.button_text = button_text
        self.form_title = form_title
        self.form_sections = form_sections
        self.sub_form_data: Optional[Dict[str, Any]] = None

    def draw(self, stdscr, y: int, width: int) -> int:
        button_display = f"[ {self.button_text} ]"

        if self.is_active:
            attr = curses.color_pair(1) | curses.A_BOLD
        else:
            attr = curses.color_pair(2)

        try:
            stdscr.addstr(y, 2, button_display, attr)
        except curses.error:
            pass

        # Show status if sub-form was completed
        if self.sub_form_data is not None:
            status = " (Completed)"
            try:
                stdscr.addstr(y, 2 + len(button_display), status)
            except curses.error:
                pass

        return 1

    def handle_key(self, key: int) -> bool:
        if not self.is_active:
            return False

        if key == ord(' ') or key == ord('\n') or key == curses.KEY_ENTER:
            # Open sub-form (this will be handled by the main form)
            return True

        return False

    def open_sub_form(self, stdscr) -> Optional[Dict[str, Any]]:
        """Open the sub-form and return its results"""
        form = Form(self.form_title, self.form_sections)
        result = form.run(stdscr)
        if result is not None:
            self.sub_form_data = result
        return result

    def get_value(self) -> Optional[Dict[str, Any]]:
        return self.sub_form_data


class Form:
    """Main form class that manages sections and input"""

    def __init__(self, title: str, sections: Dict[str, Section]):
        self.title = title
        self.sections = sections
        self.section_keys = list(sections.keys())
        self.current_section_index = 0
        self.scroll_offset = 0

    def _setup_colors(self, stdscr):
        """Initialize color pairs"""
        curses.start_color()
        curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)  # Active/selected
        curses.init_pair(2, curses.COLOR_WHITE, curses.COLOR_BLACK)  # Normal

    def _get_current_section(self) -> Section:
        """Get the currently active section"""
        if 0 <= self.current_section_index < len(self.section_keys):
            key = self.section_keys[self.current_section_index]
            return self.sections[key]
        return list(self.sections.values())[0]  # Fallback

    def _move_to_next_section(self):
        """Move to the next section"""
        self.current_section_index = (self.current_section_index + 1) % len(self.section_keys)

    def _move_to_previous_section(self):
        """Move to the previous section"""
        self.current_section_index = (self.current_section_index - 1) % len(self.section_keys)

    def _update_scroll(self, stdscr):
        """Update scroll offset to keep current section visible"""
        height, width = stdscr.getmaxyx()
        available_height = height - 4  # Reserve space for title and instructions

        # Calculate position of current section
        current_y = 0
        for i, key in enumerate(self.section_keys):
            if i == self.current_section_index:
                break
            current_y += self.sections[key].get_height() + 1  # +1 for spacing

        # Adjust scroll offset if needed
        if current_y < self.scroll_offset:
            self.scroll_offset = current_y
        elif current_y >= self.scroll_offset + available_height:
            self.scroll_offset = current_y - available_height + 1

    def _draw(self, stdscr):
        """Draw the entire form"""
        height, width = stdscr.getmaxyx()

        # Always clear the entire screen
        stdscr.erase()

        # Draw title
        title_x = max(0, (width - len(self.title)) // 2)
        try:
            stdscr.addstr(0, title_x, self.title, curses.A_BOLD | curses.A_UNDERLINE)
            stdscr.refresh()
        except curses.error:
            pass

        # Draw sections
        current_y = 2 - self.scroll_offset
        for i, key in enumerate(self.section_keys):
            section = self.sections[key]
            section.is_active = (i == self.current_section_index)

            if current_y >= 2 and current_y < height - 2:  # Within visible area
                section_height = section.draw(stdscr, current_y, width)
                current_y += section_height + 1
            else:
                current_y += section.get_height() + 1
            stdscr.refresh()

        # Draw instructions with grey background
        instructions = "Tab/Shift+Tab: Navigate | Enter: Submit | Esc: Cancel"

        # Create a full-width line with grey background
        instruction_line = " " * width
        try:
            stdscr.addstr(height - 1, 0, instruction_line, curses.A_REVERSE)
        except curses.error:
            pass

        # Center the instructions text
        if len(instructions) <= width:
            instruction_x = max(0, (width - len(instructions)) // 2)
            try:
                stdscr.addstr(height - 1, instruction_x, instructions, curses.A_REVERSE)
            except curses.error:
                pass

        stdscr.refresh()

    def run(self, stdscr) -> Optional[Dict[str, Any]]:
        """Run the form and return the results"""
        # Setup
        self._setup_colors(stdscr)
        stdscr.keypad(True)
        curses.cbreak()
        curses.noecho()
        curses.curs_set(0)

        # Set timeout to reduce CPU usage and improve responsiveness
        stdscr.timeout(50)  # 50ms timeout

        # Initial draw
        self._update_scroll(stdscr)
        self._draw(stdscr)

        while True:
            key = stdscr.getch()

            # Handle timeout (no key pressed)
            if key == -1:
                continue

            # Track if we need to redraw
            needs_redraw = False

            # Handle global keys
            if key == 27:  # Escape
                return None
            elif key == 9:  # Tab
                self._move_to_next_section()
                needs_redraw = True
            elif key == curses.KEY_BTAB:  # Shift+Tab
                self._move_to_previous_section()
                needs_redraw = True
            else:
                # Let current section handle the key first
                current_section = self._get_current_section()

                # Special handling for SubFormSection
                if isinstance(current_section, SubFormSection) and current_section.is_active:
                    if key == ord(' ') or key == ord('\n') or key == curses.KEY_ENTER:
                        # Clear screen and open sub-form
                        stdscr.erase()
                        stdscr.refresh()
                        current_section.open_sub_form(stdscr)
                        needs_redraw = True

                if not needs_redraw:
                    # Let section handle the key
                    handled = current_section.handle_key(key)

                    # If Enter key wasn't handled and it's not a sub-form, treat as submit
                    if not handled and (key == 10 or key == 13) and not isinstance(current_section, SubFormSection):
                        return self.get_value()

                    # Only redraw if the section handled the key (meaning something changed)
                    needs_redraw = handled

            # Update scroll position
            self._update_scroll(stdscr)

            # Always redraw when any key was pressed (simplest approach)
            if key != -1:
                needs_redraw = True

            # Redraw when necessary
            if needs_redraw:
                self._draw(stdscr)

    def get_value(self) -> Dict[str, Any]:
        """Get the values from all sections"""
        return {key: section.get_value() for key, section in self.sections.items()}


# Convenience function to run a form
def run_form(title: str, sections: Dict[str, Section]) -> Optional[Dict[str, Any]]:
    """Run a form in a curses environment"""
    def _run_form(stdscr):
        form = Form(title, sections)
        return form.run(stdscr)

    return curses.wrapper(_run_form)
