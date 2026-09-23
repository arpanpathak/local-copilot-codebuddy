//! Syntax highlighting for fenced code blocks, using syntect with bat's
//! collection of grammars and themes (the `two-face` crate).

use std::borrow::Cow;

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Span;
use syntect::easy::HighlightLines;
use syntect::highlighting::{FontStyle, Style as SyntectStyle};
use syntect::parsing::SyntaxSet;
use two_face::theme::{EmbeddedLazyThemeSet, EmbeddedThemeName};

const THEME: EmbeddedThemeName = EmbeddedThemeName::CatppuccinMocha;

/// Grammars and themes, loaded once at startup.
pub struct Highlighter {
    syntaxes: SyntaxSet,
    themes: EmbeddedLazyThemeSet,
}

impl Highlighter {
    /// Loads the embedded grammars and themes.
    pub fn new() -> Self {
        Self { syntaxes: two_face::syntax::extra_no_newlines(), themes: two_face::theme::extra() }
    }

    /// Starts highlighting a code block tagged `language` (`rust`, `py`, `toml`, …).
    /// Unknown languages fall back to plain text.
    pub fn code_block(&self, language: &str) -> CodeBlockHighlighter<'_> {
        let syntax = match self.syntaxes.find_syntax_by_token(language) {
            Some(syntax) => syntax,
            None => self.syntaxes.find_syntax_plain_text(),
        };
        CodeBlockHighlighter { state: HighlightLines::new(syntax, self.themes.get(THEME)), syntaxes: &self.syntaxes }
    }
}

/// Highlights the lines of one code block in order (the grammar state carries
/// over from line to line, e.g. inside a multi-line comment).
pub struct CodeBlockHighlighter<'h> {
    state: HighlightLines<'h>,
    syntaxes: &'h SyntaxSet,
}

impl CodeBlockHighlighter<'_> {
    /// Highlights one line. The spans borrow from `line` when it is borrowed.
    pub fn highlight_line<'a>(&mut self, line: Cow<'a, str>) -> Vec<Span<'a>> {
        match line {
            Cow::Borrowed(line) => self.borrowed_spans(line),
            Cow::Owned(line) => {
                let mut owned_spans = Vec::new();
                for span in self.borrowed_spans(&line) {
                    owned_spans.push(Span::styled(span.content.into_owned(), span.style));
                }
                owned_spans
            }
        }
    }

    fn borrowed_spans<'s>(&mut self, line: &'s str) -> Vec<Span<'s>> {
        let Ok(regions) = self.state.highlight_line(line, self.syntaxes) else {
            return vec![Span::raw(line)]; // grammar error: show the line unstyled
        };

        let mut spans = Vec::with_capacity(regions.len());
        for (style, text) in regions {
            spans.push(Span::styled(text, to_ratatui_style(style)));
        }
        spans
    }
}

/// Converts a syntect style (foreground colour and font flags) to ratatui.
fn to_ratatui_style(style: SyntectStyle) -> Style {
    let color = style.foreground;
    let mut result = Style::new().fg(Color::Rgb(color.r, color.g, color.b));

    if style.font_style.contains(FontStyle::BOLD) {
        result = result.add_modifier(Modifier::BOLD);
    }
    if style.font_style.contains(FontStyle::ITALIC) {
        result = result.add_modifier(Modifier::ITALIC);
    }
    if style.font_style.contains(FontStyle::UNDERLINE) {
        result = result.add_modifier(Modifier::UNDERLINED);
    }
    result
}
