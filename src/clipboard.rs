//! Copying text to the clipboard.
//!
//! On the Jetson's desktop the text goes to the system clipboard (X11 or
//! Wayland). Over SSH there is no desktop to talk to, so the text is sent to
//! the terminal instead, as an OSC 52 escape sequence, and the terminal on
//! the other end puts it on its own clipboard.

use std::io::{self, Write};

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;

/// The clipboard, kept open for the whole session: on X11 the copied text is
/// served from this process until something else is copied.
pub struct Clipboard {
    system: Option<arboard::Clipboard>,
}

impl Clipboard {
    pub fn new() -> Self {
        Self { system: arboard::Clipboard::new().ok() }
    }

    /// Copies `text`, to the system clipboard when there is one, otherwise
    /// through the terminal.
    pub fn copy(&mut self, text: &str) -> io::Result<()> {
        if let Some(system) = &mut self.system
            && system.set_text(text).is_ok()
        {
            return Ok(());
        }
        let mut stdout = io::stdout();
        write!(stdout, "\x1b]52;c;{}\x07", BASE64.encode(text))?;
        stdout.flush()
    }
}
