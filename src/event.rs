//! The events the UI thread reacts to.
//!
//! Terminal input and model output are produced on their own threads and sent
//! into one channel, so the UI loop only has to wait on that channel.

use std::sync::mpsc::Sender;
use std::thread;

use ratatui::crossterm::event;

/// Something the UI has to react to.
pub enum Event {
    /// A key press, paste, or resize from the terminal.
    Terminal(event::Event),
    /// A piece of the reply to request `request_id`.
    Reply { request_id: u64, piece: ReplyPiece },
}

/// A piece of a streamed reply.
pub enum ReplyPiece {
    /// More generated text, and how many tokens it took.
    Text { text: String, token_count: u32 },
    /// The reply is complete.
    Finished,
    /// The request failed; the message explains why.
    Failed(String),
}

/// Reads terminal events on a background thread and forwards them to `events`.
pub fn spawn_terminal_reader(events: Sender<Event>) {
    thread::spawn(move || {
        while let Ok(terminal_event) = event::read() {
            if events.send(Event::Terminal(terminal_event)).is_err() {
                break; // the UI has exited
            }
        }
    });
}
