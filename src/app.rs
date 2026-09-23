//! Application state and what each key and reply event does to it.

use std::mem;
use std::sync::mpsc::{Receiver, Sender};
use std::time::{Duration, Instant};

use anyhow::Result;
use ratatui::DefaultTerminal;
use ratatui::crossterm::event::{
    Event as TerminalEvent, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
};
use ratatui::layout::{Position, Rect};

use crate::chat::{ChatModel, Message, ReplyStream, Role};
use crate::clipboard::Clipboard;
use crate::event::{Event, ReplyPiece};
use crate::highlight::Highlighter;
use crate::{markdown, ui};

/// Rows one turn of the mouse wheel scrolls.
const WHEEL_ROWS: u16 = 3;

/// What the assistant is doing right now; shown in the status bar.
pub enum Status {
    /// Nothing sent yet.
    Idle,
    /// A reply is streaming in.
    Streaming { stream: ReplyStream, timing: ReplyTiming },
    /// The last reply finished (or was stopped).
    Finished(ReplyTiming),
    /// The last request failed.
    Failed(String),
}

/// When a reply's tokens arrived, for the speed shown in the status bar.
///
/// The wait before the first token is the model reading the prompt (prefill);
/// generation speed is measured from the first token on, so a long prompt
/// doesn't make generation look slow.
#[derive(Debug, Clone, Copy)]
pub struct ReplyTiming {
    sent_at: Instant,
    first_token_at: Option<Instant>,
    finished_at: Option<Instant>,
    pub token_count: u32,
}

impl ReplyTiming {
    fn start() -> Self {
        Self { sent_at: Instant::now(), first_token_at: None, finished_at: None, token_count: 0 }
    }

    fn add_tokens(&mut self, count: u32) {
        if self.first_token_at.is_none() && count > 0 {
            self.first_token_at = Some(Instant::now());
        }
        self.token_count += count;
    }

    fn finish(mut self) -> Self {
        self.finished_at = Some(Instant::now());
        self
    }

    /// How long the prompt took to process, once the first token has arrived.
    pub fn wait_for_first_token(&self) -> Option<Duration> {
        self.first_token_at.map(|first_token_at| first_token_at - self.sent_at)
    }

    /// Tokens per second since the first token.
    pub fn tokens_per_second(&self) -> f32 {
        let Some(first_token_at) = self.first_token_at else {
            return 0.0;
        };
        let end = self.finished_at.unwrap_or_else(Instant::now);
        let seconds = (end - first_token_at).as_secs_f32();
        self.token_count as f32 / seconds.max(0.001)
    }
}

/// Everything the chat window needs to draw itself and react to input.
pub struct App {
    pub model: ChatModel,
    pub highlighter: Highlighter,
    pub messages: Vec<Message>,
    /// Text typed into the input box, not sent yet.
    pub input: String,
    pub status: Status,
    /// First transcript row on screen, when the user has scrolled up.
    /// `None` means "stick to the bottom and follow new output".
    pub scroll_position: Option<u16>,
    /// Largest useful scroll position; updated by the UI every frame.
    pub max_scroll: u16,
    /// Rows the transcript area has on screen; updated by the UI every frame.
    pub page_height: u16,
    /// Tokens the conversation takes up so far, out of `model.context_limit()`.
    pub context_tokens: usize,
    /// Where the transcript was drawn; updated by the UI every frame.
    pub transcript_area: Rect,
    /// A short message for the status bar (e.g. "copied"), until the next key.
    pub notice: Option<String>,
    clipboard: Clipboard,
    events: Sender<Event>,
    should_quit: bool,
}

impl App {
    /// Creates the app. The optional system prompt starts every conversation.
    pub fn new(
        model: ChatModel,
        highlighter: Highlighter,
        system_prompt: Option<String>,
        events: Sender<Event>,
    ) -> Self {
        let mut messages = Vec::new();
        if let Some(prompt) = system_prompt {
            messages.push(Message { role: Role::System, content: prompt });
        }

        Self {
            model,
            highlighter,
            messages,
            input: String::new(),
            status: Status::Idle,
            scroll_position: None,
            max_scroll: 0,
            page_height: 0,
            context_tokens: 0,
            transcript_area: Rect::default(),
            notice: None,
            clipboard: Clipboard::new(),
            events,
            should_quit: false,
        }
    }

    /// Runs until the user quits: draw, wait for an event, apply it.
    pub fn run(mut self, terminal: &mut DefaultTerminal, events: &Receiver<Event>) -> Result<()> {
        while !self.should_quit {
            terminal.draw(|frame| ui::draw(frame, &mut self))?;

            let event = events.recv()?;
            self.handle_event(event);

            // Apply everything else already waiting, so a burst of tokens costs one redraw.
            while let Ok(event) = events.try_recv() {
                self.handle_event(event);
            }
        }
        Ok(())
    }

    /// Whether a reply is streaming in right now.
    pub fn is_streaming(&self) -> bool {
        matches!(self.status, Status::Streaming { .. })
    }

    /// The scroll position to draw with.
    pub fn scroll_offset(&self) -> u16 {
        match self.scroll_position {
            Some(position) => position.min(self.max_scroll),
            None => self.max_scroll,
        }
    }

    fn handle_event(&mut self, event: Event) {
        match event {
            Event::Terminal(TerminalEvent::Key(key)) if key.kind == KeyEventKind::Press => self.handle_key(key),
            Event::Terminal(TerminalEvent::Paste(text)) => self.input.push_str(&text),
            Event::Terminal(TerminalEvent::Mouse(mouse)) => self.handle_mouse(mouse),
            Event::Terminal(_) => {} // resizes just need the redraw that follows
            Event::Reply { request_id, piece } => self.handle_reply(request_id, piece),
        }
    }

    fn handle_key(&mut self, key: KeyEvent) {
        self.notice = None;
        let page = self.page_height.saturating_sub(2).max(1);

        match (key.code, key.modifiers) {
            (KeyCode::Char('d'), KeyModifiers::CONTROL) => self.should_quit = true,
            (KeyCode::Char('c'), KeyModifiers::CONTROL) => self.interrupt(),
            (KeyCode::Char('l'), KeyModifiers::CONTROL) => self.new_conversation(),
            (KeyCode::Char('y'), KeyModifiers::CONTROL) => self.copy_latest_code_block(),
            (KeyCode::Char('j'), KeyModifiers::CONTROL) => self.input.push('\n'),
            (KeyCode::Enter, KeyModifiers::ALT | KeyModifiers::SHIFT) => self.input.push('\n'),
            (KeyCode::Enter, _) => self.send_message(),
            (KeyCode::Esc, _) => self.stop_reply(),
            (KeyCode::Backspace, _) => {
                self.input.pop();
            }
            (KeyCode::Up, _) => self.scroll_up(1),
            (KeyCode::Down, _) => self.scroll_down(1),
            (KeyCode::PageUp, _) => self.scroll_up(page),
            (KeyCode::PageDown, _) => self.scroll_down(page),
            (KeyCode::End, _) => self.scroll_position = None,
            (KeyCode::Char(character), KeyModifiers::NONE | KeyModifiers::SHIFT) => self.input.push(character),
            _ => {}
        }
    }

    /// Clicking a code block's copy button copies its code; the wheel scrolls.
    fn handle_mouse(&mut self, mouse: MouseEvent) {
        match mouse.kind {
            MouseEventKind::Down(MouseButton::Left) => {
                if let Some(code) = ui::code_block_at(self, Position::new(mouse.column, mouse.row)) {
                    self.copy(&code);
                }
            }
            MouseEventKind::ScrollUp => self.scroll_up(WHEEL_ROWS),
            MouseEventKind::ScrollDown => self.scroll_down(WHEEL_ROWS),
            _ => {}
        }
    }

    /// Ctrl+Y: copies the last code block of the conversation.
    fn copy_latest_code_block(&mut self) {
        let latest = self.messages.iter().rev().find_map(|message| match message.role {
            Role::Assistant => markdown::code_blocks(&message.content).pop(),
            _ => None,
        });
        match latest {
            Some(code) => self.copy(&code),
            None => self.notice = Some("no code to copy yet".to_owned()),
        }
    }

    fn copy(&mut self, code: &str) {
        self.notice = Some(match self.clipboard.copy(code) {
            Ok(()) => format!("✓ copied {} lines", code.lines().count()),
            Err(error) => format!("cannot copy: {error}"),
        });
    }

    /// Sends the input box as a user message and starts streaming the reply.
    fn send_message(&mut self) {
        if self.is_streaming() || self.input.trim().is_empty() {
            return;
        }

        let text = mem::take(&mut self.input); // moves the text, no copy
        self.messages.push(Message { role: Role::User, content: text });

        match self.model.start_reply(&self.messages, self.events.clone()) {
            Ok(stream) => {
                self.context_tokens = stream.prompt_tokens;
                self.messages.push(Message { role: Role::Assistant, content: String::new() });
                self.status = Status::Streaming { stream, timing: ReplyTiming::start() };
                self.scroll_position = None;
            }
            Err(error) => {
                self.status = Status::Failed(format!("{error:#}"));
                self.restore_unanswered_prompt();
            }
        }
    }

    /// Applies a piece of a reply, ignoring pieces of replies that were stopped.
    fn handle_reply(&mut self, request_id: u64, piece: ReplyPiece) {
        let Status::Streaming { stream, timing } = &mut self.status else {
            return; // nothing is streaming, so this piece is left over from a stopped reply
        };
        if stream.request_id != request_id {
            return;
        }

        match piece {
            ReplyPiece::Text { text, token_count: new_tokens } => {
                if let Some(reply) = self.messages.last_mut() {
                    reply.content.push_str(&text);
                }
                timing.add_tokens(new_tokens);
                self.context_tokens += new_tokens as usize;
            }
            ReplyPiece::Finished => {
                self.status = Status::Finished(timing.finish());
                self.restore_unanswered_prompt();
            }
            ReplyPiece::Failed(error) => {
                self.status = Status::Failed(error);
                self.restore_unanswered_prompt();
            }
        }
    }

    /// Ctrl+C: stops the reply if one is streaming, otherwise clears the input.
    /// It never quits, so a stray Ctrl+C can't end the conversation.
    fn interrupt(&mut self) {
        if self.is_streaming() {
            self.stop_reply();
        } else {
            self.input.clear();
        }
    }

    /// Stops the reply that is streaming, keeping the text received so far.
    fn stop_reply(&mut self) {
        let Status::Streaming { stream, timing } = &self.status else {
            return;
        };
        stream.cancel();
        self.status = Status::Finished(timing.finish());
        self.restore_unanswered_prompt();
    }

    /// If the reply ended up empty, removes it and puts the prompt back into
    /// the input box, so the user can retry with Enter.
    fn restore_unanswered_prompt(&mut self) {
        match self.messages.last() {
            Some(reply) if reply.role == Role::Assistant && reply.content.is_empty() => {
                self.messages.pop();
            }
            Some(message) if message.role == Role::User => {} // send failed before a reply was added
            _ => return,
        }

        if let Some(prompt) = self.messages.pop_if(|message| message.role == Role::User) {
            self.input = prompt.content;
        }
    }

    /// Starts over, keeping only the system prompt.
    fn new_conversation(&mut self) {
        if let Status::Streaming { stream, .. } = &self.status {
            stream.cancel();
        }
        self.messages.retain(|message| message.role == Role::System);
        self.context_tokens = 0;
        self.status = Status::Idle;
        self.scroll_position = None;
    }

    fn scroll_up(&mut self, rows: u16) {
        self.scroll_position = Some(self.scroll_offset().saturating_sub(rows));
    }

    fn scroll_down(&mut self, rows: u16) {
        let position = self.scroll_offset().saturating_add(rows);
        // Scrolling back to the bottom resumes following new output.
        self.scroll_position = if position >= self.max_scroll { None } else { Some(position) };
    }
}
