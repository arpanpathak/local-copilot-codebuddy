//! Drawing the chat window: transcript on top, input box, status bar.

use ratatui::Frame;
use ratatui::layout::{Constraint, Flex, Layout, Margin, Position, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Paragraph, Wrap};

use crate::app::{App, ReplyTiming, Status};
use crate::chat::{Message, Role};
use crate::highlight::Highlighter;
use crate::markdown;

const ACCENT: Color = Color::Rgb(137, 180, 250);
const USER_COLOR: Color = Color::Rgb(166, 227, 161);
const MUTED: Color = Color::Rgb(108, 112, 134);
const ERROR_COLOR: Color = Color::Rgb(243, 139, 168);

/// The input box grows with its text up to this many lines.
const MAX_INPUT_LINES: u16 = 8;
const KEY_HINTS: &str = "enter send · alt+enter newline · ^y copy code · ^c stop · ^l clear · ^d quit ";

/// NVIDIA's brand green, used for the badges in the status bar.
const NVIDIA_GREEN: Color = Color::Rgb(118, 185, 0);
const BADGE_TEXT: Color = Color::Rgb(17, 17, 27);

/// Draws the whole window. Takes `&mut App` only to record the transcript's
/// scroll limits, which depend on the terminal size.
pub fn draw(frame: &mut Frame, app: &mut App) {
    let input_line_count = app.input.split('\n').count() as u16;
    let input_height = input_line_count.min(MAX_INPUT_LINES) + 2; // + top and bottom border

    let [transcript_area, input_area, status_area] =
        Layout::vertical([Constraint::Min(1), Constraint::Length(input_height), Constraint::Length(1)])
            .areas(frame.area());

    draw_transcript(frame, transcript_area.inner(Margin::new(1, 0)), app);
    draw_input(frame, input_area, app);
    draw_status_bar(frame, status_area, app);
}

/// Draws the conversation, or a greeting when it is empty.
fn draw_transcript(frame: &mut Frame, area: Rect, app: &mut App) {
    let has_conversation = app.messages.iter().any(|message| message.role != Role::System);
    if !has_conversation {
        let greeting = Line::styled(format!("chatting with {}: say something", app.model.name()), MUTED);
        let [middle_row] = Layout::vertical([Constraint::Length(1)]).flex(Flex::Center).areas(area);
        frame.render_widget(greeting.centered(), middle_row);
        return;
    }

    let (lines, _) = transcript_lines(&app.messages, &app.highlighter);
    let transcript = Paragraph::new(lines).wrap(Wrap { trim: false });

    let total_rows = transcript.line_count(area.width) as u16;
    app.max_scroll = total_rows.saturating_sub(area.height);
    app.page_height = area.height;
    app.transcript_area = area;

    frame.render_widget(transcript.scroll((app.scroll_offset(), 0)), area);
}

/// A code block's copy button in the transcript.
struct CopyButton {
    /// The transcript line the button is on.
    line: usize,
    /// The message the code block belongs to.
    message: usize,
    /// Which code block of that message it is.
    block: usize,
}

/// The code of the block whose copy button is at `position` on screen, if any.
pub fn code_block_at(app: &App, position: Position) -> Option<String> {
    let area = app.transcript_area;
    if !area.contains(position) {
        return None;
    }
    let (lines, buttons) = transcript_lines(&app.messages, &app.highlighter);

    // Find the transcript line on the clicked row, allowing for wrapped lines.
    let clicked_row = usize::from(app.scroll_offset() + position.y - area.y);
    let mut row = 0;
    let mut clicked_line = None;
    for (index, line) in lines.into_iter().enumerate() {
        row += Paragraph::new(line).wrap(Wrap { trim: false }).line_count(area.width);
        if row > clicked_row {
            clicked_line = Some(index);
            break;
        }
    }

    let button = buttons.iter().find(|button| Some(button.line) == clicked_line)?;
    markdown::code_blocks(&app.messages[button.message].content).into_iter().nth(button.block)
}

/// Turns the conversation into styled lines, and notes where the copy buttons
/// are. The lines borrow the message text; nothing is copied.
fn transcript_lines<'a>(messages: &'a [Message], highlighter: &Highlighter) -> (Vec<Line<'a>>, Vec<CopyButton>) {
    let user_header = Style::new().fg(USER_COLOR).add_modifier(Modifier::BOLD);
    let assistant_header = Style::new().fg(ACCENT).add_modifier(Modifier::BOLD);

    let mut lines = Vec::new();
    let mut buttons = Vec::new();
    for (message_index, message) in messages.iter().enumerate() {
        match message.role {
            Role::System => continue, // the system prompt is not shown
            Role::User => {
                lines.push(Line::styled("❯ you", user_header));
                for text_line in message.content.lines() {
                    lines.push(Line::raw(text_line));
                }
            }
            Role::Assistant => {
                lines.push(Line::styled("◆ assistant", assistant_header));
                let rendered = markdown::render(&message.content, highlighter);
                for (block, header) in rendered.code_headers.into_iter().enumerate() {
                    buttons.push(CopyButton { line: lines.len() + header, message: message_index, block });
                }
                lines.extend(rendered.lines);
            }
        }
        lines.push(Line::default()); // blank line between messages
    }
    (lines, buttons)
}

/// Draws the input box and places the cursor at the end of the text.
fn draw_input(frame: &mut Frame, area: Rect, app: &App) {
    let border_color = if app.is_streaming() { MUTED } else { ACCENT };
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(border_color)
        .title(Span::styled(" message ", MUTED));
    let inner = block.inner(area);

    // The cursor sits after the last character, so keep that end in view.
    let row_count = app.input.split('\n').count() as u16;
    let last_line = app.input.rsplit('\n').next().unwrap_or_default();
    let cursor_column = Span::raw(last_line).width() as u16;
    let scroll_x = (cursor_column + 1).saturating_sub(inner.width);
    let scroll_y = row_count.saturating_sub(inner.height);

    let input = Paragraph::new(app.input.as_str()).block(block).scroll((scroll_y, scroll_x));
    frame.render_widget(input, area);

    if !app.is_streaming() {
        let cursor_x = inner.x + cursor_column - scroll_x;
        let cursor_y = inner.y + row_count - 1 - scroll_y;
        frame.set_cursor_position(Position::new(cursor_x, cursor_y));
    }
}

/// Draws the model name and progress on the left, key hints on the right.
fn draw_status_bar(frame: &mut Frame, area: Rect, app: &App) {
    let progress = match &app.status {
        Status::Idle => Span::raw(""),
        Status::Streaming { timing, .. } => match timing.wait_for_first_token() {
            None => Span::styled("reading the prompt…", ACCENT),
            Some(_) => Span::styled(format!("generating · {}", reply_speed(timing)), ACCENT),
        },
        Status::Finished(timing) => Span::styled(reply_speed(timing), MUTED),
        Status::Failed(error) => Span::styled(error.as_str(), ERROR_COLOR),
    };
    // Badges: a solid green "NVIDIA" block, then "TensorRT-LLM" in green on dark.
    let nvidia_badge = Style::new().fg(BADGE_TEXT).bg(NVIDIA_GREEN).add_modifier(Modifier::BOLD);
    let tensorrt_badge = Style::new().fg(NVIDIA_GREEN).bg(BADGE_TEXT).add_modifier(Modifier::BOLD);
    let status = Line::from(vec![
        Span::styled(" ▲ NVIDIA ", nvidia_badge),
        Span::raw(" "),
        Span::styled(" TensorRT-LLM ", tensorrt_badge),
        Span::styled(format!(" {} ", app.model.name()), ACCENT),
        Span::styled(format!("ctx {}/{} · ", app.context_tokens, app.model.context_limit()), MUTED),
        progress,
    ]);

    // The status keeps its full width and the key hints give way on narrow
    // terminals, but a notice ("copied") always shows in full.
    let status_width = status.width() as u16;
    let (right, right_constraint) = match &app.notice {
        Some(notice) => {
            let notice = Line::styled(format!(" {notice} "), USER_COLOR);
            let width = notice.width() as u16;
            (notice, Constraint::Length(width))
        }
        None => (Line::styled(KEY_HINTS, MUTED), Constraint::Fill(1)),
    };
    let [status_area, right_area] =
        Layout::horizontal([Constraint::Max(status_width), right_constraint]).flex(Flex::SpaceBetween).areas(area);
    frame.render_widget(status, status_area);
    frame.render_widget(right.right_aligned(), right_area);
}

/// Formats a reply's speed, e.g. `212 tok · 16.3 tok/s · first token 0.4s`.
fn reply_speed(timing: &ReplyTiming) -> String {
    let speed = format!("{} tok · {:.1} tok/s", timing.token_count, timing.tokens_per_second());
    match timing.wait_for_first_token() {
        Some(wait) => format!("{speed} · first token {:.1}s", wait.as_secs_f32()),
        None => speed,
    }
}
