//! Turns Markdown into styled terminal lines.
//!
//! The spans borrow from the source text, so rendering copies no message
//! content. Half-finished Markdown (a reply still streaming in) is fine: an
//! unclosed code fence simply runs to the end of the text.

use std::borrow::Cow;
use std::mem;

use pulldown_cmark::{CodeBlockKind, CowStr, Event, HeadingLevel, Options, Parser, Tag, TagEnd};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

use crate::highlight::{CodeBlockHighlighter, Highlighter};

const ACCENT: Color = Color::Rgb(137, 180, 250);
const MUTED: Color = Color::Rgb(108, 112, 134);
const MARGIN_STYLE: Style = Style::new().fg(MUTED);
const INLINE_CODE_STYLE: Style = Style::new().fg(Color::Rgb(250, 179, 135));
const LINK_STYLE: Style = Style::new().fg(ACCENT).add_modifier(Modifier::UNDERLINED);
const LIST_MARKER_STYLE: Style = Style::new().fg(ACCENT);
/// The copy button on a code block's header: dark text on the accent colour.
const COPY_BUTTON_STYLE: Style = Style::new().fg(Color::Rgb(17, 17, 27)).bg(ACCENT).add_modifier(Modifier::BOLD);
/// The label of the copy button on every code block.
pub const COPY_BUTTON: &str = " copy ";

const HORIZONTAL_RULE: &str = "────────────────────────────────────────";
const CODE_GUTTER: &str = "▏ ";
const QUOTE_BAR: &str = "│ ";
/// Source of indentation slices for list items (`"1. "` → 3 spaces).
const SPACES: &str = "          ";

const OPTIONS: Options = Options::ENABLE_STRIKETHROUGH.union(Options::ENABLE_TABLES).union(Options::ENABLE_TASKLISTS);

/// Markdown rendered into terminal lines.
pub struct Rendered<'a> {
    pub lines: Vec<Line<'a>>,
    /// The index in `lines` of each code block's header (the line with its
    /// copy button), in the same order as [`code_blocks`] returns the code.
    pub code_headers: Vec<usize>,
}

/// Renders `source` into lines that borrow from it.
pub fn render<'a>(source: &'a str, highlighter: &Highlighter) -> Rendered<'a> {
    let mut renderer = MarkdownRenderer::new(highlighter);
    for event in Parser::new_ext(source, OPTIONS) {
        renderer.handle_event(event);
    }
    renderer.finish()
}

/// The code of each code block in `source`, as it would be pasted.
pub fn code_blocks(source: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut current: Option<String> = None;
    for event in Parser::new_ext(source, OPTIONS) {
        match event {
            Event::Start(Tag::CodeBlock(_)) => current = Some(String::new()),
            Event::Text(text) if current.is_some() => current.as_mut().unwrap().push_str(&text),
            Event::End(TagEnd::CodeBlock) => blocks.extend(current.take()),
            _ => {}
        }
    }
    blocks.extend(current); // a block still streaming in
    blocks
}

/// Walks pulldown-cmark's event stream and builds lines.
struct MarkdownRenderer<'a, 'h> {
    highlighter: &'h Highlighter,
    /// Finished lines.
    lines: Vec<Line<'a>>,
    /// Where each code block's header is in `lines`.
    code_headers: Vec<usize>,
    /// The line being built.
    current_line: Vec<Span<'a>>,
    /// Nested inline styles (bold inside a link inside a heading…).
    style_stack: Vec<Style>,
    /// Nested left margins (list indentation, quote bars, the code gutter).
    margins: Vec<&'static str>,
    /// One entry per open list: the next item number, or `None` for bullets.
    open_lists: Vec<Option<u64>>,
    /// Set while inside a fenced or indented code block.
    code_block: Option<CodeBlockHighlighter<'h>>,
}

impl<'a, 'h> MarkdownRenderer<'a, 'h> {
    fn new(highlighter: &'h Highlighter) -> Self {
        Self {
            highlighter,
            lines: Vec::new(),
            code_headers: Vec::new(),
            current_line: Vec::new(),
            style_stack: Vec::new(),
            margins: Vec::new(),
            open_lists: Vec::new(),
            code_block: None,
        }
    }

    /// Returns the lines, without trailing blank ones.
    fn finish(mut self) -> Rendered<'a> {
        self.end_line_if_not_empty();
        while self.lines.last().is_some_and(|line| line.spans.is_empty()) {
            self.lines.pop();
        }
        Rendered { lines: self.lines, code_headers: self.code_headers }
    }

    fn handle_event(&mut self, event: Event<'a>) {
        match event {
            Event::Start(tag) => self.start_tag(tag),
            Event::End(tag) => self.end_tag(tag),
            Event::Text(text) if self.code_block.is_some() => self.add_code(text),
            Event::Text(text) | Event::InlineMath(text) | Event::FootnoteReference(text) => {
                self.add_span(Span::styled(text, self.current_style()));
            }
            Event::Code(code) | Event::InlineHtml(code) => self.add_span(Span::styled(code, INLINE_CODE_STYLE)),
            Event::Html(text) | Event::DisplayMath(text) => self.add_verbatim(text),
            Event::SoftBreak => self.add_span(Span::raw(" ")),
            Event::HardBreak => self.end_line(),
            Event::Rule => {
                self.end_line_if_not_empty();
                self.add_span(Span::styled(HORIZONTAL_RULE, MARGIN_STYLE));
                self.end_block();
            }
            Event::TaskListMarker(done) => {
                let checkbox = if done { "[x] " } else { "[ ] " };
                self.add_span(Span::styled(checkbox, LIST_MARKER_STYLE));
            }
        }
    }

    fn start_tag(&mut self, tag: Tag<'a>) {
        match tag {
            Tag::Heading { level, .. } => {
                self.end_line_if_not_empty();
                self.add_span(Span::styled(heading_marker(level), MARGIN_STYLE));
                self.style_stack.push(heading_style(level));
            }
            Tag::CodeBlock(kind) => self.start_code_block(kind),
            Tag::List(first_number) => {
                self.end_line_if_not_empty();
                self.open_lists.push(first_number);
            }
            Tag::Item => self.start_list_item(),
            Tag::BlockQuote(_) => {
                self.end_line_if_not_empty();
                self.margins.push(QUOTE_BAR);
                self.push_style(Style::new().fg(MUTED).add_modifier(Modifier::ITALIC));
            }
            Tag::Emphasis => self.push_style(Style::new().add_modifier(Modifier::ITALIC)),
            Tag::Strong => self.push_style(Style::new().add_modifier(Modifier::BOLD)),
            Tag::Strikethrough => self.push_style(Style::new().add_modifier(Modifier::CROSSED_OUT)),
            Tag::Link { .. } => self.push_style(LINK_STYLE),
            Tag::TableHead => self.push_style(Style::new().add_modifier(Modifier::BOLD)),
            Tag::TableCell if !self.current_line.is_empty() => self.add_span(Span::styled(" │ ", MARGIN_STYLE)),
            _ => {}
        }
    }

    fn end_tag(&mut self, tag: TagEnd) {
        match tag {
            TagEnd::Paragraph | TagEnd::Table => self.end_block(),
            TagEnd::Heading(_) => {
                self.style_stack.pop();
                self.end_block();
            }
            TagEnd::CodeBlock => {
                self.end_line_if_not_empty();
                self.code_block = None;
                self.margins.pop();
                self.end_block();
            }
            TagEnd::List(_) => {
                self.open_lists.pop();
                let was_outermost_list = self.open_lists.is_empty();
                if was_outermost_list {
                    self.end_block();
                }
            }
            TagEnd::Item => {
                self.end_line_if_not_empty();
                self.margins.pop();
            }
            TagEnd::BlockQuote(_) => {
                self.end_line_if_not_empty();
                self.margins.pop();
                self.style_stack.pop();
                self.end_block();
            }
            TagEnd::TableHead => {
                self.style_stack.pop();
                self.end_line_if_not_empty();
            }
            TagEnd::TableRow => self.end_line_if_not_empty(),
            TagEnd::Emphasis | TagEnd::Strong | TagEnd::Strikethrough | TagEnd::Link => {
                self.style_stack.pop();
            }
            _ => {}
        }
    }

    /// Opens a code block: a gutter on the left, and on top the language and a copy button.
    fn start_code_block(&mut self, kind: CodeBlockKind<'a>) {
        self.end_line_if_not_empty();

        // The info string can carry extras, e.g. "rust,ignore" or "py title=x".
        let language = match &kind {
            CodeBlockKind::Fenced(info) => info.split([',', ' ']).next().unwrap_or_default(),
            CodeBlockKind::Indented => "",
        };
        self.code_block = Some(self.highlighter.code_block(language));
        self.margins.push(CODE_GUTTER);

        if !language.is_empty() {
            let label_style = MARGIN_STYLE.add_modifier(Modifier::ITALIC);
            self.add_span(Span::styled(language.to_owned(), label_style));
            self.add_span(Span::raw("  "));
        }
        self.add_span(Span::styled(COPY_BUTTON, COPY_BUTTON_STYLE));
        self.code_headers.push(self.lines.len());
        self.end_line();
    }

    /// Starts a list item with its bullet or number, and indents what follows.
    fn start_list_item(&mut self) {
        self.end_line_if_not_empty();

        let marker = match self.open_lists.last_mut() {
            Some(Some(next_number)) => {
                let marker = format!("{next_number}. ");
                *next_number += 1;
                Span::styled(marker, LIST_MARKER_STYLE)
            }
            _ => Span::styled("• ", LIST_MARKER_STYLE),
        };

        let indent_width = marker.width().min(SPACES.len());
        self.add_span(marker);
        self.margins.push(&SPACES[..indent_width]);
    }

    /// Adds highlighted code, one output line per source line.
    fn add_code(&mut self, text: CowStr<'a>) {
        let Some(mut highlighter) = self.code_block.take() else { return };
        for source_line in split_lines(text) {
            for span in highlighter.highlight_line(source_line) {
                self.add_span(span);
            }
            self.end_line();
        }
        self.code_block = Some(highlighter);
    }

    /// Adds raw text (HTML blocks, display math) line by line, unformatted.
    fn add_verbatim(&mut self, text: CowStr<'a>) {
        for source_line in split_lines(text) {
            self.add_span(Span::styled(source_line, self.current_style()));
            self.end_line();
        }
    }

    fn current_style(&self) -> Style {
        self.style_stack.last().copied().unwrap_or_default()
    }

    /// Layers `style` on top of the current style, until the matching end tag.
    fn push_style(&mut self, style: Style) {
        let combined = self.current_style().patch(style);
        self.style_stack.push(combined);
    }

    /// Adds a span to the current line, starting it with the margins if it is new.
    fn add_span(&mut self, span: Span<'a>) {
        if self.current_line.is_empty() {
            self.add_margins();
        }
        self.current_line.push(span);
    }

    fn add_margins(&mut self) {
        for margin in &self.margins {
            self.current_line.push(Span::styled(*margin, MARGIN_STYLE));
        }
    }

    /// Finishes the current line, even when empty (blank lines inside code matter).
    fn end_line(&mut self) {
        if self.current_line.is_empty() {
            self.add_margins();
        }
        let line = mem::take(&mut self.current_line);
        self.lines.push(Line::from(line));
    }

    /// Finishes the current line if anything was added to it.
    fn end_line_if_not_empty(&mut self) {
        if !self.current_line.is_empty() {
            self.end_line();
        }
    }

    /// Finishes a block (paragraph, heading, list…) and leaves one blank line after it.
    fn end_block(&mut self) {
        self.end_line_if_not_empty();
        let last_line_is_blank = self.lines.last().is_none_or(|line| line.spans.is_empty());
        if !last_line_is_blank {
            self.lines.push(Line::default());
        }
    }
}

/// Splits text into lines, borrowing when the text is borrowed from the source
/// (which it is, except for rare cases where the parser had to rewrite it).
fn split_lines(text: CowStr<'_>) -> Vec<Cow<'_, str>> {
    match text {
        CowStr::Borrowed(text) => text.lines().map(Cow::Borrowed).collect(),
        rewritten => rewritten.lines().map(|line| Cow::Owned(line.to_owned())).collect(),
    }
}

/// The `#` prefix shown before a heading.
fn heading_marker(level: HeadingLevel) -> &'static str {
    match level {
        HeadingLevel::H1 => "# ",
        HeadingLevel::H2 => "## ",
        HeadingLevel::H3 => "### ",
        HeadingLevel::H4 => "#### ",
        HeadingLevel::H5 => "##### ",
        HeadingLevel::H6 => "###### ",
    }
}

fn heading_style(level: HeadingLevel) -> Style {
    let bold = Style::new().add_modifier(Modifier::BOLD);
    match level {
        HeadingLevel::H1 => bold.fg(Color::Rgb(203, 166, 247)),
        HeadingLevel::H2 => bold.fg(ACCENT),
        _ => bold,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The text of each line, without styles.
    fn plain_text(lines: &[Line<'_>]) -> Vec<String> {
        lines.iter().map(|line| line.to_string()).collect()
    }

    #[test]
    fn renders_common_blocks() {
        let source = "# Title\n\nSome **bold** and `code`.\n\n- one\n- two\n\n1. first\n2. second\n\n> quoted\n";
        let lines = render(source, &Highlighter::new()).lines;
        let expected =
            ["# Title", "", "Some bold and code.", "", "• one", "• two", "", "1. first", "2. second", "", "│ quoted"];
        assert_eq!(plain_text(&lines), expected);
    }

    #[test]
    fn highlights_code_and_borrows_from_the_source() {
        let source = "```rust\nfn main() {}\n\n```\n";
        let rendered = render(source, &Highlighter::new());
        let lines = rendered.lines;
        assert_eq!(plain_text(&lines), ["▏ rust   copy ", "▏ fn main() {}", "▏ "]);
        assert_eq!(rendered.code_headers, [0]);

        let code_spans = &lines[1].spans[1..]; // skip the gutter
        assert!(code_spans.len() > 1, "expected several highlighted tokens");
        assert!(code_spans.iter().all(|span| matches!(span.content, Cow::Borrowed(_))));
        assert!(code_spans.iter().any(|span| span.style.fg.is_some()));
    }

    #[test]
    fn code_blocks_hold_the_code_without_markup() {
        let source =
            "Try:\n\n```rust\nfn main() {\n    run();\n}\n```\n\nthen\n\n```\nls -la\n```\n\n```sh\necho still stre";
        let rendered = render(source, &Highlighter::new());
        let blocks = code_blocks(source);
        assert_eq!(blocks, ["fn main() {\n    run();\n}\n", "ls -la\n", "echo still stre"]);
        assert_eq!(rendered.code_headers.len(), blocks.len());
        for header in rendered.code_headers {
            assert!(rendered.lines[header].to_string().contains(COPY_BUTTON));
        }
    }

    #[test]
    fn unclosed_fence_mid_stream_still_renders() {
        let lines = render("text\n\n```py\ndef f(", &Highlighter::new()).lines;
        assert_eq!(plain_text(&lines).last().map(String::as_str), Some("▏ def f("));
    }
}
